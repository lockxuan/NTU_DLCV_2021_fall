

FILENAME='./p1_model.pth'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uYrWhLwppJZgw-X9hH_vF7o8lrrcu6oy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uYrWhLwppJZgw-X9hH_vF7o8lrrcu6oy" -O $FILENAME && rm -rf /tmp/cookies.txt



python3 p1/inference.py \
		--test_path $1 \
		--output_csv $2 \
		--model_path $FILENAME