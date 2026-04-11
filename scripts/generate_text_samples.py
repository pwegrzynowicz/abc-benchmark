from abc_benchmark.generation.feature_text_generator import make_generator

gen = make_generator("medium")
scene = gen.generate(seed=42, difficulty_name="medium")
print(scene.prompt)
print("gold:", scene.gold_label)