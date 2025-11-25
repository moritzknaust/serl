# Description:
#   SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning

load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")
load("//tools/build_defs/license:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//googlex/giza/rtt:__subpackages__"],
)

license(name = "license")

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "serl_launcher",
    srcs = glob(
        ["**/*.py"],
        exclude = [
            "tests/**",
        ],
    ),
    deps = [
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/agentlace",
        "//third_party/py/chex",
        "//third_party/py/distrax",
        "//third_party/py/dm_env",
        "//third_party/py/dm_env:specs",
        "//third_party/py/einops",
        "//third_party/py/flax",
        "//third_party/py/flax:core",
        "//third_party/py/gym:gym_internal",
        "//third_party/py/gymnasium:gymnasium_internal",
        "//third_party/py/imageio",
        "//third_party/py/jax",
        "//third_party/py/lz4:frame",
        "//third_party/py/ml_collections",
        "//third_party/py/numpy",
        "//third_party/py/optax",
        "//third_party/py/requests",
        "//third_party/py/scipy",
        "//third_party/py/setuptools",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_datasets",
        "//third_party/py/tensorflow_probability",
        "//third_party/py/tqdm",
    ],
)

py_test(
    name = "serl_launcher_test",
    srcs = ["tests/serl_launcher_test.py"],
    deps = [
        ":serl_launcher",
    ],
)
