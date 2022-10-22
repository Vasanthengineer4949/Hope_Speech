import config
from model import Model

if __name__ == "__main__":
    tam_model = Model(config.TAM_TRAIN_EMB_PATH, config.TAM_TEST_EMB_PATH, lang="tam")
    tam_model.run()
    kan_model = Model(config.KAN_TRAIN_EMB_PATH, config.KAN_TEST_EMB_PATH, lang="kan")
    kan_model.run()
    eng_model = Model(config.ENG_TRAIN_EMB_PATH, config.ENG_TEST_EMB_PATH, lang="eng")
    eng_model.run()
    span_model = Model(config.SPAN_TRAIN_EMB_PATH, config.SPAN_TEST_EMB_PATH, lang="span")
    span_model.run()