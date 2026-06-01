#! /bin/bash
model=$1
wait_time=10
# dataset=('Adiac' 'CinCECGTorso' 'CricketX' 'CricketY' 'CricketZ' 'DiatomSizeReduction' 
# 'FacesUCR' 'FiftyWords' 'FordB' 'Lightning2' 'Lightning7' 
# 'Mallat' 'MedicalImages' 'RefrigerationDevices' 'ScreenType' 'ShapeletSim' 'SmallKitchenAppliances' 
# 'SonyAIBORobotSurface1' 'SonyAIBORobotSurface2' 'StarLightCurves' 'ToeSegmentation1'
#  'ToeSegmentation2' 'UWaveGestureLibraryAll' 'UWaveGestureLibraryX' 'UWaveGestureLibraryY'
#  'UWaveGestureLibraryZ')

dataset=('Adiac' 'CricketX' 'FordB')
hiddens=(1 2 3 4 5)

failed=()
for data in ${dataset[@]}
do
    for hidden in ${hiddens[@]}
    do
        if ! python train1.py --dataset $data --gpu 2 --att 6 --ratio $hidden --log_dir ./log/${data}_ratio${hidden}/; then
            failed+=("$data") 
        fi

        if [ $wait_time -gt 0 ]; then  
            sleep $wait_time  
        fi  
    done
done

if [ ${#failed[@]} -gt 0 ]; then  
    echo "Failed to train the following datasets:"  
    for data in ${failed[@]}; do
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "以下数据集训练失败："  
    for f in "${failed[@]}"; do  
        echo "$f"  
    done  
fi