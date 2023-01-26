"""Tests for AutoResetWrapper.

Taken from seals/tests/test_wrappers.py"""
from seals import util
from seals.testing import envs


def test_auto_reset_wrapper_pad(episode_length=3, n_steps=100, n_manual_reset=2):
    """This test also exists in seals. The advantage of also having it here is that
    if we decide to update our version of seals this test will show us whether there
    were any changes in the parts of seals that we care about.

    Check that AutoResetWrapper returns correct values from step and reset.
    AutoResetWrapper that pads trajectory with an extra transition containing the
    terminal observations.
    Also check that calls to .reset() do not interfere with automatic resets.
    Due to the padding the number of steps counted inside the environment and the number
    of steps performed outside the environment, i.e. the number of actions performed,
    will differ. This test checks that this difference is consistent.
    """
    env = util.AutoResetWrapper(
        envs.CountingEnv(episode_length=episode_length),
        discard_terminal_observation=False,
    )

    for _ in range(n_manual_reset):
        obs = env.reset()
        assert obs == 0

        # We count the number of episodes, so we can sanity check the padding.
        num_episodes = 0
        next_episode_end = episode_length
        for t in range(1, n_steps + 1):
            act = env.action_space.sample()
            obs, rew, done, info = env.step(act)

            # AutoResetWrapper overrides all done signals.
            assert done is False

            if t == next_episode_end:
                # Unlike the AutoResetWrapper that discards terminal observations,
                # here the final observation is returned directly, and is not stored
                # in the info dict.
                # Due to padding, for every episode the final observation is offset from
                # the outer step by one.
                assert obs == (t - num_episodes) / (num_episodes + 1)
                assert rew == episode_length * 10
            if t == next_episode_end + 1:
                num_episodes += 1
                # Because the final step returned the final observation, the initial
                # obs of the next episode is returned in this additional step.
                assert obs == 0
                # Consequently, the next episode end is one step later, so it is
                # episode_length steps from now.
                next_episode_end = t + episode_length

                # Reward of the 'reset transition' is fixed to be 0.
                assert rew == 0

                # Sanity check padding. Padding should be 1 for each past episode.
                assert (
                    next_episode_end
                    == (num_episodes + 1) * episode_length + num_episodes
                )
