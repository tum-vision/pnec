while getopts d:r:v:s:i:n:gt: flag
do
    case "${flag}" in
        d) dataset_path=${OPTARG};;
        r) results_path=${OPTARG};;
        v) visualization_path=${OPTARG};;
        s) starting_value=${OPTARG};;
        i) iterations=${OPTARG};;
        n) no_skip=${OPTARG};;
        gt) use_ground_trutn${OPTARG};;
    esac
done
: "${dataset_path:="{default location of sequences e.g. [...]/kitti/sequences/}"}"
: "${results_path:="{default location of results}"}"
: "${visualization_path:="$results_path/visualization/"}"
: "${starting_value:=0}"
: "${iterations:="10"}"
: "${no_skip:="true"}"
: "${use_ground_truth:="false"}"

[ ! -d "$visualization_path" ] && mkdir -p "$visualization_path"
tracking_config_path="data/tracking/KITTI"
tracking_calib_path="data/tracking/KITTI/kitti_calib.json"
config_path="data"
pnec_config="${config_path}/test_config.yaml"

sequence_nums=(0 1 2 3 4 5 6 7 8 9 10)
sequences=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10")
let end_v=$iterations+$starting_value-1
for sequence_num in "${sequence_nums[@]}"
do
    sequence=${sequences[sequence_num]}
    tracking_path="${tracking_config_path}/${sequence}.json"
    images_path="${dataset_path}/${sequence}/image_0"
    timestamp_path="${dataset_path}/${sequence}/times.txt"
    gt_path="${dataset_path}/${sequence}/poses.txt"
    gt=""

    if [ ${use_ground_truth} == "true" ]
    then
        gt="-gt=${gt_path}"
    fi
    # Copy ground truth into results folder
    cp $gt_path $results_path/$sequence/poses.txt
    cp $timestamp_path $results_path/$sequence/times.txt

    if [ $sequence == "00" ] || [ $sequence == "01" ] || [ $sequence == "02" ]
    then
        camera_config="$config_path/config_kitti00-02.yaml"
    fi
    if [ $sequence == "03" ]
    then
        camera_config="$config_path/config_kitti03.yaml"
    fi
    if [ $sequence == "04" ] || [ $sequence == "05" ] || [ $sequence == "06" ] || [ $sequence == "07" ] || [ $sequence == "08" ] || [ $sequence == "09" ] || [ $sequence == "10" ]
    then
        camera_config="$config_path/config_kitti04-10.yaml"
    fi

    for i in $(seq $starting_value $end_v);
    do
        [ ! -d "$results_path/$sequence" ] && mkdir -p "$results_path/$sequence"
        [ ! -d "$results_path/$sequence/$i" ] && mkdir -p "$results_path/$sequence/$i"
 
        numbered_output="$results_path/$sequence/$i/"

        ./build/pnec_vo $camera_config $pnec_config $tracking_path $tracking_calib_path $images_path $timestamp_path $numbered_output $visualization_path $no_skip ${gt} &
    done
    wait
    python3 scripts/sequence_evaluation.py -d $results_path -s ${sequence}
done