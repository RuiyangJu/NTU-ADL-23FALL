mkdir ./multiple-choice/model
mkdir ./question-answering/model
mkdir ./multiple-choice/output

gdown 1DSXtdsLhw8cp2J7yrQL41DAZX-EVDcuS -O ./multiple-choice/model/model_mc.zip
gdown 1KChE3Jo_6ORK-SsJtUq3vGjSUyFCTDMU -O ./question-answering/model/model_qa.zip

unzip ./multiple-choice/model/model_mc.zip -d ./multiple-choice/model
unzip ./question-answering/model/model_qa.zip -d ./question-answering/model
