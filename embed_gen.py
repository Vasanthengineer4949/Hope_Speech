import config
from embed import Embedder

if __name__ == "__main__":
    tam_train_embedder = Embedder(
                            config.TAM_TRAIN_PATH,
                            "tam",
                            config.MODEL_CKPT,
                            "train")
    tam_train_embedder.run()
    tam_test_embedder = Embedder(
                            config.TAM_TEST_PATH,
                            "tam",
                            config.MODEL_CKPT,
                            "test")
    tam_test_embedder.run()

    kan_train_embedder = Embedder(
                            config.KAN_TRAIN_PATH,
                            "kan",
                            config.MODEL_CKPT,
                            "train")
    kan_train_embedder.run()
    kan_test_embedder = Embedder(
                            config.KAN_TEST_PATH,
                            "kan",
                            config.MODEL_CKPT,
                            "test")
    kan_test_embedder.run()

    eng_train_embedder = Embedder(
                            config.ENG_TRAIN_PATH,
                            "eng",
                            config.MODEL_CKPT,
                            "train")
    eng_train_embedder.run()
    eng_test_embedder = Embedder(
                            config.ENG_TEST_PATH,
                            "eng",
                            config.MODEL_CKPT,
                            "test")
    eng_test_embedder.run()

    span_train_embedder = Embedder(
                            config.SPAN_TRAIN_PATH,
                            "span",
                            config.MODEL_CKPT,
                            "train")
    span_train_embedder.run()
    span_test_embedder = Embedder(
                            config.SPAN_TEST_PATH,
                            "span",
                            config.MODEL_CKPT,
                            "test")
    span_test_embedder.run()