#!/bin/bash


conda activate xy_py38

cd /home/bhui/ML/xiaoyang/Graduation_Design/MyProject/


python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=attractive
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=atypical
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=boring
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=calm
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=caring
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=cold
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=common
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=confident
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=egotistic
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=emotional
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=emotStable
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=emotUnstable
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=familiar
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=forgettable
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=friendly
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=happy
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=humble
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=intelligent
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=interesting
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=introverted
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=irresponsible
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=kind
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=mean
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=memorable
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=normal
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=responsible
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=sociable
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=trustworthy
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=typical
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=unattractive
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=uncertain
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=uncommon
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=unemotional
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=unfamiliar
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=unfriendly
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=unhappy
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=unintelligent
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=untrustworthy
python main.py --model=VIT --batch_size=40 --lr=1e-4 --target=weird


wait
