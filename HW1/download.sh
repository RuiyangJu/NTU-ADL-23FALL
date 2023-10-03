mkdir ./data
mkdir ./multiple-choice/model

gdown 1Pgbvr3CYRUT8fSwmjszdPdqu_C73vCu0 -O ./data/all_data.zip
gdown 1e9ggtDDt6Ri6hnsU5CSbBPH7ux3tIQoh -O ./multiple-choice/model/trained_model.zip

unzip ./data/all_data.zip -d ./data
unzip ./multiple-choice/model/trained_model.zip -d ./multiple-choice/model
