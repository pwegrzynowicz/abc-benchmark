from abc_benchmark.generation.cluster_generator import make_generator


def test_gold_label_matches_target_count():
    gen = make_generator("easy")
    scene = gen.generate(seed=321, difficulty_name="easy")
    counted = sum(
        1
        for item in scene.items
        if item.cluster_id == scene.target_cluster_id
        and not item.is_anchor
        and item.shape == scene.target_shape
        and item.color == scene.target_color
    )
    assert counted == scene.gold_label
