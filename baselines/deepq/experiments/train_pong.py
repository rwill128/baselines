from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari


import tensorflow
tensorflow.compat.v1.disable_eager_execution()

def main():
    logger.configure()
    env = make_atari('PongNoFrameskip-v4')
    env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)

    model = deepq.learn(
        env=env,
        network="conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        lr=1e-4,
        total_timesteps=int(1e5),
        print_freq=1,
        checkpoint_freq=1,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=100,
        target_network_update_freq=100,
        gamma=0.99,
    )

    model.save('pong_model.pkl')
    env.close()


if __name__ == '__main__':
    main()
