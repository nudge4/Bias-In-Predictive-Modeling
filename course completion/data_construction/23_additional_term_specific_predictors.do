forvalues y=2014/2019 {
	foreach v in "3_spe" "4_sue" "6_fae" {
		preserve
			zipuse "C:\\Users\\ys8mz\\Box Sync\\VCCS restricted student data\\Build\Class\\Class_`y'_`v'.dta.zip", clear
			keep vccsid strm college subject course_num credit day_eve_code
			duplicates drop
			drop if credit == 0
			gen dev_flag = length(course_num) < 3
			bys vccsid strm: egen dev = max(dev_flag)
			gen lvl200_flag = (length(course_num) == 3 & substr(course_num, 1, 1) == "2")
			drop dev_flag
			egen day_eve_code_new = mode(day_eve_code), by(vccsid strm college subject course_num) maxmode
			drop day_eve_code
			gen course = subject + "_" + course_num
			drop subject course_num
			collapse (sum) credit, by(vccsid strm college course dev day_eve_code_new lvl200_flag)
			isid vccsid strm college course
			gen credit_200 = credit * lvl200_flag
			egen sum_credit = sum(credit), by(vccsid strm)
			egen sum_credit_200 = sum(credit_200), by(vccsid strm)
			gen lvl2_share = sum_credit_200/sum_credit
			drop credit_200 sum_credit_200 lvl200_flag
			gen eve_flag = (day_eve_code == "E")
			gen credit_eve = credit * eve_flag
			egen sum_credit_eve = sum(credit_eve), by(vccsid strm)
			gen eve_share = sum_credit_eve/sum_credit
			drop eve_flag credit_eve sum_credit_eve
			gen online_flag = (day_eve_code == "O")
			gen credit_online = credit * online_flag
			egen sum_credit_online = sum(credit_online), by(vccsid strm)
			gen online_share = sum_credit_online/sum_credit
			drop online_flag credit_online sum_credit_online sum_credit
			tempfile tmp
			save "`tmp'", replace
		restore
		append using "`tmp'", force
	}
}
drop day_eve_code_new
preserve
	keep vccsid strm dev *_share
	duplicates drop
	sort vccsid strm
	merge 1:1 vccsid strm using "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\all_vccsid_strm.dta", keep(2 3) nogen
	drop first_strm
	sort vccsid strm
	save "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\additional_term_specific.dta", replace
restore
merge m:1 vccsid strm using "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\all_vccsid_strm.dta", keep(2 3) nogen
drop dev *_share first_strm
sort vccsid strm college course
bys vccsid strm: gen course_counts = _N
drop if course_counts == 1
drop course_counts
sort vccsid strm college course
save "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\concurrent_courses.dta", replace
