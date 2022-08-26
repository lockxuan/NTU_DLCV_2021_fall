

mnist_train_path='./hw2_data/digits/mnistm/train'
mnist_test_path='./hw2_data/digits/mnistm/test'

svhn_train_path='./hw2_data/digits/svhn/train'
svhn_test_path='./hw2_data/digits/svhn/test'

usps_train_path='../hw2_data/digits/usps/train'
usps_test_path='./hw2_data/digits/usps/test'



python p3/p3_train_source.py \
				--train_path $svhn_train_path \
				--test_path $mnist_test_path \
				--model_name p3_1_s2m

python p3/p3_train_source.py \
				--train_path $mnist_train_path \
				--test_path $usps_test_path \
				--model_name p3_1_m2u

python p3/p3_train_source.py \
				--train_path $usps_train_path \
				--test_path $svhn_test_path \
				--model_name p3_1_u2s

