from abc_benchmark.generation.cluster_generator import make_generator


def test_generator_produces_scene():
    gen = make_generator("easy")
    scene = gen.generate(seed=123, difficulty_name="easy")
    assert scene.gold_label >= 0
    assert len(scene.items) > 0
    assert "Respond with a number only" in scene.prompt
