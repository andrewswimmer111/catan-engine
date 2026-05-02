import rl
import rl.env
import rl.encoding
import rl.agents
import rl.models
import rl.training
import rl.evaluation
import rl.replay
import rl.utils


def test_submodules_importable():
    for mod in (rl, rl.env, rl.encoding, rl.agents, rl.models,
                rl.training, rl.evaluation, rl.replay, rl.utils):
        assert mod is not None
