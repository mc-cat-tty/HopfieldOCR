#!/bin/sh

python3 -m manim -pqh scenes/intro.py &
python3 -m manim -pqh scenes/infra1.py &
python3 -m manim -pqh scenes/infra2.py &
python3 -m manim -pqh scenes/infra3.py &
python3 -m manim -pqh scenes/demo_nn.py &
python3 -m manim -pqh scenes/demo_ocr.py --disable_caching &
python3 -m manim -pqh scenes/nn_scene.py &
python3 -m manim -pqk scenes/energy_landscape.py &
python3 -m manim -pqh scenes/outro.py
