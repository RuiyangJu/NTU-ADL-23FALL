mkdir ./data
mkdir ./multiple-choice/model
mkdir ./question-answering/model

gdown 1Pgbvr3CYRUT8fSwmjszdPdqu_C73vCu0 -O ./data/all_data.zip
gdown 13clU4Y_7dl5nv8ae23Br1dRGo0X7DhcJ -O ./multiple-choice/output/predict_mc.zip
# gdown 1DSXtdsLhw8cp2J7yrQL41DAZX-EVDcuS -O ./multiple-choice/model/model_mc.zip
gdown 1KChE3Jo_6ORK-SsJtUq3vGjSUyFCTDMU -O ./question-answering/model/model_qa.zip

unzip ./data/all_data.zip -d ./data
unzip ./multiple-choice/output/predict_mc.zip -d ./multiple-choice/output
# unzip ./multiple-choice/model/model_mc.zip -d ./multiple-choice/model
unzip ./question-answering/model/model_qa.zip -d ./question-answering/model
