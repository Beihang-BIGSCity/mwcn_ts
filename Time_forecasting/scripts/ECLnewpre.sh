for hd in 6 8 10 12 14 16
do
	for lr in 1e-3 5e-4 2e-4
	do	
		echo "hidden_size:$hd lr:$lr"
		python3 -u train2.py --log_dir='ECLnew1' --hidden_size=$hd --lr=$lr 
	done
done 

