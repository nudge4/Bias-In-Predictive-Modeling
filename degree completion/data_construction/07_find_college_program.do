/* This script identifies the college and program (in terms of both CIP and CURR) for each observation, which will be used in both college-specific & program-specific models as well as exploratory analysis of the effects of college and program on demographic biases. */

clear
forvalues y=2007/2017 {
	foreach t in "3_spe" "4_sue" "6_fae" {
		preserve
			zipuse "D:\Yifeng -- Project Work\ys8mz_sandbox\Build\Student\Student_`y'_`t'.dta.zip", clear
			* drop if curr_degree == "N/A"
			keep vccsid strm college cip curr total_credit_hrs curr_degree
			gen cip_2digit = floor(cip)
			drop cip
			gen degree_lvl = substr(curr_degree,1,1)
			gsort vccsid degree_lvl -total_credit_hrs
			bys vccsid: keep if _n == 1
			drop curr_degree degree_lvl total_credit_hrs
			isid vccsid
			sort vccsid
			rename cip_2digit cip
			tempfile stu
			save "`stu'", replace
		restore
		append using "`stu'", force
	}
}
drop if strm == 2072
rename strm last_term
sort vccsid last_term
isid vccsid last_term

merge 1:1 vccsid last_term using "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\full_data_truncated.dta", keepusing(first_nonde_strm) keep(2 3) nogen
drop first_nonde_strm
drop if college == ""
replace cip = 0 if cip == .

bys cip: gen cip_count = _N
bys curr: gen curr_count = _N

replace cip = 0 if cip_count < _N * 0.001
replace curr = "000" if substr(curr, 1, 1) == "0" | curr_count < _N * 0.001

drop cip_count curr_count last_term
sort vccsid

save "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\bias\student_college_program.dta", replace
