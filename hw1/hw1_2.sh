	

FILENAME='./p2_model.pth'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_I0nyZjlIK62HT4QlsGTY_AXhLf1hOK0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_I0nyZjlIK62HT4QlsGTY_AXhLf1hOK0" -O $FILENAME && rm -rf /tmp/cookies.txt



python3 p2/inference.py \
		--test_path $1 \
		--output_path $2 \
		--model_path $FILENAME