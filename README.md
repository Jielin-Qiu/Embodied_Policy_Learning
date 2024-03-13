# Embodied Executable Policy Learning with Language-based Scene Summarization

[Embodied Executable Policy Learning with Language-based Scene Summarization](https://arxiv.org/abs/2306.05696). In NAACL 2024


## Citation

If you feel our code or models help your research, kindly cite our papers:

```
@@article{Qiu2023EmbodiedEP,
  title={Embodied Executable Policy Learning with Language-based Scene Summarization},
  author={Jielin Qiu and Mengdi Xu and William Jongwon Han and Seungwhan Moon and Ding Zhao},
  booktitle={NAACL},
  year={2024}
}
}
```


## Dataset
The curated dataset can be found[here](https://drive.google.com/drive/folders/1gCmE61eg-Bbt7ZL0pZOC2_477xPwrDO7?usp=sharing)

## Preprocessing
Preprocessing scripts are in preprocess_utils.py, preprocess_prompt.py, and preprocess_video.py

## Set up environment
Create a virtual environment and activate it. 

```
python -m venv .env
source .env/bin/activate
```

Install basic requirements.

```
pip install -r requirements.txt
```

## Configurations
All customizable configurations are in schema.py

## Finetuning/Inference
To finetune or evaluate the SUM or APM model, please see main.py and add your desired arguments. 
You can also choose your desired learning paradigm (supervised/REINFORCE) in main.py.
