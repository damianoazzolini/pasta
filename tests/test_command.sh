# $1 -> test name
# $2 -> command
# $3 -> expected output
function run_test {
    echo -n "Running $1: "
	ntest=$((ntest+1))
	# to manage empty result
	res=$3
	if [ "${#res}" -eq 0 ]; then
		diff -w <(python3 ../src/paspsp/paspsp.py $2) <(echo -e -n $3)
	else
    	diff -w <(python3 ../src/paspsp/paspsp.py $2) <(echo -e $3)
    fi
	# echo $?
    if [ "$?" -eq 0 ]; then
	    echo -e "Success"
	else
        echo -e "\033[91m*** Failed ***\e[0m"
		failed=$((failed+1))
    fi
}

run_test "bird_2_2_fly_1" "../examples/bird_2_2.lp --query=\"fly_1\"" "Lower probability for the query fly_1: 0.600\nUpper probability for the query fly_1: 0.700"

run_test "bird_4_fly(1)" "../examples/bird_4.lp --query=\"fly(1)\"" "Lower probability for the query fly(1): 0.250\nUpper probability for the query fly(1): 0.500"

run_test "bird_4_nofly(1)" "../examples/bird_4.lp --query=\"nofly(1)\"" "Lower probability for the query nofly(1): 0\nUpper probability for the query nofly(1): 0.250"

run_test "bird_10_fly(1)" "../examples/bird_10.lp --query=\"fly(1)\"" "Lower probability for the query fly(1): 0.127\nUpper probability for the query fly(1): 0.500"

run_test "bird_10_nofly(1)" "../examples/bird_10.lp --query=\"nofly(1)\"" "Lower probability for the query nofly(1): 0\nUpper probability for the query nofly(1): 0.373"

run_test "path_path_1_4" "../examples/path.lp --query=\"path(1,4)\"" "Lower probability == upper probability for the query path(1,4): 0.267"

run_test "viral_marketing_5_buy_5" "../examples/viral_marketing_5.lp --query=\"buy(5)\"" "Lower probability for the query buy(5): 0.273\nUpper probability for the query buy(5): 0.290"