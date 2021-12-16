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
        echo -e "\033[92mSuccess\e[0m"
	else
        echo -e "\033[91m*** Failed ***\e[0m"
		failed=$((failed+1))
    fi
}

run_test "bird_2_2_fly_1" "../examples/bird_2_2.lp --query=\"fly_1\"" "Lower probability for the query fly_1: 0.6\nUpper probability for the query fly_1: 0.7"

run_test "bird_4_fly(1)" "../examples/bird_4.lp --query=\"fly(1)\"" "Lower probability for the query fly(1): 0.25\nUpper probability for the query fly(1): 0.5"

run_test "bird_4_different_fly(1)" "../examples/bird_4_different.lp --query=\"fly(1)\"" "Lower probability for the query fly(1): 0.102222\nUpper probability for the query fly(1): 0.11"

run_test "bird_4_nofly(1)" "../examples/bird_4.lp --query=\"nofly(1)\"" "Lower probability for the query nofly(1): 0.0\nUpper probability for the query nofly(1): 0.25"

run_test "bird_10_fly(1)" "../examples/bird_10.lp --query=\"fly(1)\"" "Lower probability for the query fly(1): 0.126953\nUpper probability for the query fly(1): 0.5"

run_test "bird_10_nofly(1)" "../examples/bird_10.lp --query=\"nofly(1)\"" "Lower probability for the query nofly(1): 0.0\nUpper probability for the query nofly(1): 0.373046"

run_test "path_path_1_4" "../examples/path.lp --query=\"path(1,4)\"" "Lower probability == upper probability for the query path(1,4): 0.266815"

run_test "viral_marketing_5_buy_5" "../examples/viral_marketing_5.lp --query=\"buy(5)\"" "Lower probability for the query buy(5): 0.2734\nUpper probability for the query buy(5): 0.29"

run_test "bird_4_different_q_fly_1_e_fly_2" "../examples/bird_4_different.lp --evidence=\"fly(2)\"" "Lower probability for the query: 0.073952\nUpper probability for the query: 0.113255"

run_test "bird_4_cond_q_fly_1" "../examples/conditionals/bird_4_cond.lp --query=\"fly\"" "Lower probability for the query fly: 0.7\nUpper probability for the query fly: 1.0"

run_test "smokers_cond_q_smk" "../examples/conditionals/smokers.lp --query=\"smk\"" "Lower probability for the query smk: 0.7\nUpper probability for the query smk: 0.70627"

run_test "sick_sick" "../examples/sick.lp" "Lower probability for the query: 0.199\nUpper probability for the query: 0.2374"

run_test "disjunction" "../examples/disjunction.lp" "Lower probability for the query: 0.6\nUpper probability for the query: 0.8"