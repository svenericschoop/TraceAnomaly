# -*- coding: utf-8 -*-
import functools

import os, time
import click
import tensorflow as tf
import pandas as pd
from tensorflow.contrib.framework import arg_scope, add_arg_scope
from sklearn.model_selection import train_test_split
import gc
import psutil

import tfsnippet as spt
from tfsnippet.examples.utils import (print_with_title,
                                      collect_outputs,
                                      MultiGPU)
from traceanomaly.readdata_custom import get_data_vae_custom, get_data_vae_unsupervised, print_data_summary
from traceanomaly.MLConfig import (MLConfig,
                       global_config as config,
                       config_options)

class ExpConfig(MLConfig):
    debug_level = -1  # -1: disable all checks;
                      #  0: assertions only
                      #  1: assertions + numeric check

    # model parameters
    z_dim = 10
    x_dim = 100
    
    flow_type = None # None: no flow
                       # planar_nf:
                       # rnvp
    n_planar_nf_layers = 10
    n_rnvp_layers = 10
    n_rnvp_hidden_layers = 1

    # training parameters
    write_summary = False
    max_epoch = 2000
    max_step = None
    batch_size = 256
    
    l2_reg = 0.0001
    initial_lr = 0.001
    
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 100
    lr_anneal_step_freq = None

    # evaluation parameters
    test_n_z = 500
    test_batch_size = 128

    norm_clip = 10

    # GPU configuration
    use_gpu = True
    gpu_memory_fraction = 0.8  # Use 80% of GPU memory


@spt.global_reuse
@add_arg_scope
def q_net(x, posterior_flow, observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)

    # compute the hidden features
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_x = tf.to_float(x)
        h_x = spt.layers.dense(h_x, 500)
        h_x = spt.layers.dense(h_x, 500)

    # sample z ~ q(z|x)
    z_mean = spt.layers.dense(h_x, config.z_dim, name='z_mean')
    z_std = 1e-4 + tf.nn.softplus(
        spt.layers.dense(h_x, config.z_dim, name='z_std'))
    z = net.add('z', spt.Normal(mean=z_mean, std=z_std), n_samples=n_z,
                group_ndims=1, flow=posterior_flow)

    return net


@spt.global_reuse
@add_arg_scope
def p_net(observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)

    # sample z ~ p(z)
    z = net.add('z', spt.Normal(mean=tf.zeros([1, config.z_dim]),
                                logstd=tf.zeros([1, config.z_dim])),
                group_ndims=1, n_samples=n_z)

    # compute the hidden features
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_z = z
        h_z = spt.layers.dense(h_z, 500)
        h_z = spt.layers.dense(h_z, 500)

    # sample x ~ p(x|z)
    x_mean = spt.layers.dense(h_z, config.x_dim, name='x_mean')
    x_std = 1e-4 + tf.nn.softplus(
        spt.layers.dense(h_z, config.x_dim, name='x_std'))
    x = net.add('x', spt.Normal(mean=x_mean, std=x_std), group_ndims=1)

    return net


def coupling_layer_shift_and_scale(x1, n2):
    # compute the hidden features
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h = x1
        for _ in range(config.n_rnvp_hidden_layers):
            h = spt.layers.dense(h, 500)

    # compute shift and scale
    shift = spt.layers.dense(
        h, n2, kernel_initializer=tf.zeros_initializer(),
        bias_initializer=tf.zeros_initializer(), name='shift'
    )
    scale = spt.layers.dense(
        h, n2, kernel_initializer=tf.zeros_initializer(),
        bias_initializer=tf.zeros_initializer(), name='scale'
    )
    return shift, scale


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


@click.command()
@click.option('--data_dir', help='The path to the processed data directory', metavar='PATH',
              required=True, type=str)
@click.option('--outputpath',
              help='The name of output files',
              metavar='PATH',
              required=True, type=str)
@click.option('--max_samples', help='Maximum number of samples to load (for memory efficiency)', 
              metavar='INT', type=int, default=None)
@click.option('--sample_rate', help='Fraction of data to sample (0.1 = 10%)', 
              metavar='FLOAT', type=float, default=1.0)
@click.option('--use_gpu', help='Enable GPU usage', 
              is_flag=True, default=True)
@click.option('--gpu_memory_fraction', help='Fraction of GPU memory to use (0.0-1.0)', 
              metavar='FLOAT', type=float, default=0.8)
@config_options(ExpConfig)
def main(data_dir, outputpath, max_samples, sample_rate, use_gpu, gpu_memory_fraction):
    # Update config with command line parameters
    config.use_gpu = use_gpu
    config.gpu_memory_fraction = gpu_memory_fraction
    
    if config.debug_level == -1:
        spt.utils.set_assertion_enabled(False)
    elif config.debug_level == 1:
        spt.utils.set_check_numerics(True)

    # print the config
    print_with_title('Configurations', config.format_config(), after='\n')
    
    # Monitor initial memory usage
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.1f} MB")

    # Print data summary
    print_data_summary(data_dir)

    # Check if test files exist for supervised vs unsupervised training
    test_normal_file = os.path.join(data_dir, 'test_normal')
    test_abnormal_file = os.path.join(data_dir, 'test_abnormal')
    
    if os.path.exists(test_normal_file) and os.path.exists(test_abnormal_file):
        # Supervised training with test data
        print("Found test files, using supervised training mode")
        (x_train, y_train), (x_test, y_test), flows_test = \
            get_data_vae_custom(data_dir)
        unsupervised_mode = False
    else:
        # Unsupervised training without test data
        print("Test files not found, using unsupervised training mode")
        x_train, flows_train = get_data_vae_unsupervised(data_dir, max_samples, sample_rate)
        y_train = None  # No labels in unsupervised mode
        x_test = None
        y_test = None
        flows_test = None
        unsupervised_mode = True
    
    config.x_dim = x_train.shape[1]

    all_len = x_train.shape[0]
    print('origin data: %s' % all_len)
    print('Feature dimension: %s' % config.x_dim)
    
    # Monitor memory after data loading
    data_loaded_memory = get_memory_usage()
    print(f"Memory after data loading: {data_loaded_memory:.1f} MB (delta: {data_loaded_memory - initial_memory:.1f} MB)")
    
    # Force garbage collection
    gc.collect()
    gc_memory = get_memory_usage()
    print(f"Memory after garbage collection: {gc_memory:.1f} MB")
    
    # Split training data for validation
    valid_rate = 0.1
    x_train, x_valid = train_test_split(x_train, test_size=valid_rate)
    
    print('%s for validation, %s for training' % (x_valid.shape[0], x_train.shape[0]))
    if not unsupervised_mode:
        print('%s for test' % x_test.shape[0])
    else:
        print('No test data (unsupervised mode)')
    print('x_dim: %s z_dim: %s' % (config.x_dim, config.z_dim))

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.float32, shape=(None, config.x_dim), name='input_x')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)

    # build the posterior flow
    if config.flow_type is None:
        posterior_flow = None
    elif config.flow_type == 'planar_nf':
        posterior_flow = \
            spt.layers.planar_normalizing_flows(config.n_planar_nf_layers)
    else:
        assert(config.flow_type == 'rnvp')
        with tf.variable_scope('posterior_flow'):
            flows = []
            for i in range(config.n_rnvp_layers):
                flows.append(spt.layers.ActNorm())
                flows.append(spt.layers.CouplingLayer(
                    tf.make_template(
                        'coupling',
                        coupling_layer_shift_and_scale,
                        create_scope_now_=True
                    ),
                    scale_type='sigmoid'
                ))
                flows.append(spt.layers.InvertibleDense(strict_invertible=True))
            posterior_flow = spt.layers.SequentialFlow(flows=flows)

    # derive the initialization op
    with tf.name_scope('initialization'), \
            arg_scope([spt.layers.act_norm], initializing=True):
        # Use GPU if available
        device = '/device:GPU:0' if config.use_gpu else '/device:CPU:0'
        with tf.device(device):
            init_q_net = q_net(input_x, posterior_flow)
            init_chain = init_q_net.chain(
                p_net, latent_axis=0, observed={'x': input_x})
            init_loss = tf.reduce_mean(init_chain.vi.training.sgvb())

    # derive the loss and lower-bound for training
    with tf.name_scope('training'):
        # Use GPU if available
        device = '/device:GPU:0' if config.use_gpu else '/device:CPU:0'
        with tf.device(device):
            train_q_net = q_net(input_x, posterior_flow)
            train_chain = train_q_net.chain(
                p_net, latent_axis=0, observed={'x': input_x})

            vae_loss = tf.reduce_mean(train_chain.vi.training.sgvb())
            loss = vae_loss + tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        # Use GPU if available
        device = '/device:GPU:0' if config.use_gpu else '/device:CPU:0'
        with tf.device(device):
            test_q_net = q_net(input_x, posterior_flow, n_z=config.test_n_z)
            test_chain = test_q_net.chain(
                p_net, latent_axis=0, observed={'x': input_x})
            test_logp = test_chain.vi.evaluation.is_loglikelihood()
            test_nll = -tf.reduce_mean(test_logp)
            test_lb = tf.reduce_mean(test_chain.vi.lower_bound.elbo())

    # derive the optimizer
    with tf.name_scope('optimizing'):
        # Use GPU if available
        device = '/device:GPU:0' if config.use_gpu else '/device:CPU:0'
        with tf.device(device):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            params = tf.trainable_variables()
            grads = optimizer.compute_gradients(loss, var_list=params)

            cliped_grad = []
            for grad, var in grads:
                if grad is not None and var is not None:
                    if config.norm_clip is not None:
                        grad = tf.clip_by_norm(grad, config.norm_clip)
                    cliped_grad.append((grad, var))

            with tf.control_dependencies(
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.apply_gradients(cliped_grad)

    train_flow = spt.DataFlow.arrays([x_train],
                                     config.batch_size,
                                     shuffle=True,
                                     skip_incomplete=True)
    valid_flow = spt.DataFlow.arrays([x_valid],
                                     config.test_batch_size)
    
    if not unsupervised_mode:
        test_flow = spt.DataFlow.arrays([x_test],
                                        config.test_batch_size)
    else:
        test_flow = None

    # model_file
    model_name = os.path.join(
        'models',
        'md_{}_{}.model'.format(
            config.flow_type or 'vae',
            outputpath
        )
    )

    # Create output directory
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Monitor memory before TensorFlow session creation
    pre_tf_memory = get_memory_usage()
    print(f"Memory before TensorFlow session: {pre_tf_memory:.1f} MB")
    
    # Configure GPU usage
    if config.use_gpu:
        # Check if GPU is available using TensorFlow 1.x API
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        gpu_devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
        if gpu_devices:
            print(f"Found {len(gpu_devices)} GPU(s): {gpu_devices}")
            print(f"Using GPU device: {gpu_devices[0]}")
        else:
            print("No GPU devices found, falling back to CPU")
            config.use_gpu = False
    else:
        print("GPU usage disabled in configuration")
    
    # Print device information
    device_info = "GPU" if config.use_gpu else "CPU"
    print(f"Training will use: {device_info}")
    
    # Create memory-efficient session configuration
    session_config = tf.ConfigProto()
    if config.use_gpu:
        session_config.gpu_options.allow_growth = True
        if hasattr(config, 'gpu_memory_fraction'):
            session_config.gpu_options.per_process_gpu_memory_fraction = config.gpu_memory_fraction
    else:
        session_config.device_count['GPU'] = 0  # Force CPU usage
    
    with spt.utils.create_session(lock_memory=False, **session_config).as_default() as session:
        var_dict = spt.utils.get_variables_as_dict()
        saver = spt.VariableSaver(var_dict, model_name)
        
        if os.path.exists(model_name):
            print('%s exists, loading model...' % model_name)
            saver.restore()
        else:
            print('No model found, starting training...')
            # initialize the network
            spt.utils.ensure_variables_initialized()
            for [batch_x] in train_flow:
                print('Network initialization loss: {:.6g}'.
                      format(session.run(init_loss, {input_x: batch_x})))
                print('')
                break

            # train the network
            with spt.TrainLoop(params,
                               var_groups=['p_net', 'q_net', 'posterior_flow'],
                               max_epoch=config.max_epoch,
                               max_step=config.max_step,
                               early_stopping=True,
                               valid_metric_name='valid_loss',
                               valid_metric_smaller_is_better=True) as loop:
                trainer = spt.Trainer(
                    loop, train_op, [input_x], train_flow,
                    metrics={'loss': loss}
                )
                trainer.anneal_after(
                    learning_rate,
                    epochs=config.lr_anneal_epoch_freq,
                    steps=config.lr_anneal_step_freq
                )
                evaluator = spt.Evaluator(
                    loop,
                    metrics={'valid_loss': test_nll},
                    inputs=[input_x],
                    data_flow=valid_flow,
                    time_metric_name='valid_time'
                )

                trainer.evaluate_after_epochs(evaluator, freq=10)
                trainer.log_after_epochs(freq=1)
                trainer.run()
            saver.save()

        # get the answer
        if not unsupervised_mode:
            print('Starting testing...')
            start = time.time()
            test_ans = collect_outputs([test_logp], [input_x], test_flow)[0] \
                / config.x_dim
            end = time.time()
            print("Test time: ", end-start)
            
            # Save results
            output_file = os.path.join('results', '{}_{}.csv'.format(config.flow_type or 'vae', outputpath))
            valid_file = os.path.join('results', 'v{}_{}.csv'.format(config.flow_type or 'vae', outputpath))
            
            pd.DataFrame(
                {'id': flows_test, 'label': y_test, 'score': test_ans}) \
                .to_csv(output_file, index=False)
            print(f'Test results saved to: {output_file}')
        else:
            print('Skipping testing (unsupervised mode)')
            valid_file = os.path.join('results', 'v{}_{}.csv'.format(config.flow_type or 'vae', outputpath))
        
        # Always save validation results
        valid_ans = collect_outputs([test_logp], [input_x], valid_flow)[0] \
            / config.x_dim
        pd.DataFrame({'score': valid_ans}).to_csv(valid_file, index=False)
        print(f'Validation results saved to: {valid_file}')


if __name__ == '__main__':
    main()
