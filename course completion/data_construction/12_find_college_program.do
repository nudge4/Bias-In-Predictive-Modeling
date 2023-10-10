clear
forvalues y=2014/2018 {
	foreach t in "3_spe" "4_sue" "6_fae" {
		preserve
			zipuse "D:\Yifeng -- Project Work\ys8mz_sandbox\Build\Student\Student_`y'_`t'.dta.zip", clear
			* drop if curr_degree == "N/A"
			keep vccsid strm college cip curr total_credit_hrs curr_degree
			gen cip_2digit = floor(cip)
			drop cip
			gen degree_lvl = substr(curr_degree, 1, 1)
			gsort vccsid college degree_lvl -total_credit_hrs
			bys vccsid college: keep if _n == 1
			drop curr_degree total_credit_hrs
			isid vccsid college
			sort vccsid college
			rename cip_2digit cip
			tempfile stu
			save "`stu'", replace
		restore
		append using "`stu'", force
	}
}
sort vccsid strm college
isid vccsid strm college

merge 1:m vccsid strm college using "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\train_df.dta", keepusing(course section) keep(2 3) nogen
replace cip = 0 if cip == .
save "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\train_college_program.dta", replace


clear
forvalues y=2019/2019 {
	foreach t in "3_spe" "4_sue" "6_fae" {
		preserve
			zipuse "D:\Yifeng -- Project Work\ys8mz_sandbox\Build\Student\Student_`y'_`t'.dta.zip", clear
			* drop if curr_degree == "N/A"
			keep vccsid strm college cip curr total_credit_hrs curr_degree
			gen cip_2digit = floor(cip)
			drop cip
			gen degree_lvl = substr(curr_degree, 1, 1)
			gsort vccsid college degree_lvl -total_credit_hrs
			bys vccsid college: keep if _n == 1
			drop curr_degree total_credit_hrs
			isid vccsid college
			sort vccsid college
			rename cip_2digit cip
			tempfile stu
			save "`stu'", replace
		restore
		append using "`stu'", force
	}
}
sort vccsid strm college
isid vccsid strm college

merge 1:m vccsid strm college using "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\test_df.dta", keepusing(course section) keep(2 3) nogen
replace cip = 0 if cip == .
replace degree_lvl = "N" if degree_lvl == ""
replace curr = "000" if mi(curr)
save "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\test_college_program.dta", replace


clear
use "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\train_college_program.dta"
append using "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\test_college_program.dta", force

bys cip: gen cip_count = _N
bys curr: gen curr_count = _N
replace cip = 0 if cip_count < _N * 0.001
replace curr = "000" if substr(curr, 1, 1) == "0" | curr_count < _N * 0.001

drop cip_count curr_count
sort vccsid strm college course section

save "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\student_college_program.dta", replace
