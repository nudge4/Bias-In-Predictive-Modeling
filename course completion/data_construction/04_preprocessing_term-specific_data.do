*****************************************************************************************
** First deal with the missing grades issue (Same as the first part in "preprocessing.do"
*****************************************************************************************
** There are courses with multiple sessions within the same student x college x term: if one session has a non-missing grade but the rest of the sessions have missing grades,
** then we can fill in those missing grades with the grade of the non-missing session of the course
use "D:\Yifeng -- Project Work\ys8mz_sandbox\Merged_Class.dta", clear
merge m:1 vccsid using "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\all_vccsid.dta", keep(3) nogen
drop if strm >= 2202


replace course_num = substr(course_num, 2, .) if regexm(course_num,"^[0-9][0-9][0-9][0-9]L$")
replace course_num = substr(course_num, 1, 3) if regexm(course_num,"^[0-9][0-9][0-9]L$")
sort vccsid college strm subject course_num
gen tmp = 1 if grade == ""
egen flag = max(tmp), by(vccsid college strm subject course_num) // flag indicates the course has missing grades
egen num_of_grades_tmp = nvals(grade), by(vccsid college strm subject course_num) 
egen num_of_grades = max(num_of_grades_tmp), by(vccsid college strm subject course_num) // if there's a unique non-missing grade for the course
drop num_of_grades_tmp
bys vccsid college strm subject course_num: gen num_of_obs = _N
egen new_grade = mode(grade) if flag == 1 & num_of_grades == 1, by(vccsid college strm subject course_num)
replace grade = new_grade if tmp == 1 & new_grade != "" // fill in the missing values under this category
drop new_grade tmp flag num_of_grades
drop if credit == 0 & num_of_obs == 1 & mi(grade)
drop num_of_obs
** There are courses with multiple sessions within the same student x term: if one session has a non-missing grade but the rest of the sessions have missing grades,
** then we can fill in those missing grades with the grade of the non-missing session of the course
sort vccsid strm subject course_num
gen tmp = 1 if grade == ""
egen flag = max(tmp), by(vccsid strm subject course_num)
egen num_of_grades_tmp = nvals(grade), by(vccsid strm subject course_num)
egen num_of_grades = max(num_of_grades_tmp), by(vccsid strm subject course_num)
drop num_of_grades_tmp
bys vccsid strm subject course_num: gen num_of_obs = _N
egen new_grade = mode(grade) if flag == 1 & num_of_grades == 1, by(vccsid strm subject course_num)
replace grade = new_grade if tmp == 1 & new_grade != ""
drop new_grade tmp flag num_of_obs num_of_grades
preserve // For each actively enrolled term, identify number of prior enrolled terms at VCCS (For each student)
	keep vccsid strm
	duplicates drop
	sort vccsid strm
	isid vccsid strm
	bys vccsid: gen num_of_prior_terms = _n - 1
	gen su_tmp = mod(strm, 10) == 3
	bys vccsid: gen num_of_prior_su_terms = sum(su_tmp)
	replace num_of_prior_su_terms = num_of_prior_su_terms - 1 if mod(strm, 10) == 3
	sort vccsid strm
	tempfile term_num
	save "`term_num'", replace
restore


****************************************************************************************************************************************************************
** Next find out the term GPA and term enrollment intensity by term for each historical VCCS student (the trendline predictors will be constructed using Python)
****************************************************************************************************************************************************************
preserve // The procedure is similar as what's used in "processing.do"
	gen adj_credit = credit
	replace adj_credit = credit/3 if inlist(grade, "W") // adj_credit is used for calculating term_enrl_intesity
	replace adj_credit = credit/2 if inlist(grade, "X", "XY", "XN", "R") // we assume enrollment intensity of audited courses as well as the courses marked as "re-enroll" is only worth 1/2 of the credits
	gen effective_credit_2 = credit // effective_credits_2 is used for estimating the running term gpa
	replace effective_credit_2 = 0 if !inlist(grade, "A", "B", "C", "D", "F")
	gen numeric_grade = 0
	replace numeric_grade = 4 if grade == "A"
	replace numeric_grade = 3 if grade == "B"
	replace numeric_grade = 2 if grade == "C"
	replace numeric_grade = 1 if grade == "D"
	gen grade_point_2 = numeric_grade*effective_credit_2
	sort vccsid strm subject course_num
	bys vccsid strm: egen term_credits_attempted = sum(adj_credit)
	bys vccsid strm: egen sum_grade_point_2 = sum(grade_point_2)
	bys vccsid strm: egen sum_effective_credit_2 = sum(effective_credit_2)
	gen term_gpa = sum_grade_point_2/sum_effective_credit_2
	keep vccsid strm term_credits_attempted term_gpa
	duplicates drop
	sort vccsid strm
	isid vccsid strm
	bys vccsid: gen term_num = _n
	replace term_credits_attempted = 1.5*term_credits_attempted if mod(strm, 10) == 3 // adjust the enrollment intensity for Summer to get a more coherent estimation of trendline of enrollment intensity predictors
	save "C:\\Users\\ys8mz\\Box Sync\\Clickstream\\data\\bias\\term_lvl_gpa_enrl_intensity.dta", replace
	keep vccsid strm
	duplicates drop
	sort vccsid strm
	bys vccsid: keep if _n == 1
	rename strm first_strm
	save "C:\\Users\\ys8mz\\Box Sync\\Clickstream\\data\\bias\\first_strm.dta", replace
restore


*************************************************************************************************************
** Finally construct four additional predictors: num_of_prior_terms, pct_withdrawn, pct_incomplete, audit_ind
*************************************************************************************************************
merge m:1 vccsid strm using "`term_num'", keep(3) nogen
merge m:1 vccsid using "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\first_strm.dta", keep(3) nogen
drop su_tmp
gen x = num_of_prior_terms - num_of_prior_su_terms
gen y = .
replace y = (strm-first_strm)/5 if mod(strm,10) == mod(first_strm,10)
replace y = (strm-first_strm+1)/5 - 1 if mod(strm-first_strm, 10) == 9 & mod(first_strm, 10) == 3
replace y = (strm-first_strm+1)/5 if mod(strm-first_strm, 10) == 9 & mod(first_strm, 10) == 4
replace y = (strm-first_strm+2)/5 - 1 if mod(strm-first_strm, 10) == 8
replace y = (strm-first_strm-1)/5 if mod(strm-first_strm, 10) == 1 & mod(first_strm, 10) == 3
replace y = (strm-first_strm-1)/5 + 1 if mod(strm-first_strm, 10) == 1 & mod(first_strm, 10) == 2
replace y = (strm-first_strm-2)/5 + 1 if mod(strm-first_strm, 10) == 2
assert x <= y
gen pct_stopped = (y-x)/y
drop x y
/*
gen tmp = 0
replace tmp = 1 if inlist(grade, "X", "XN", "XY") // audited courses
sort vccsid strm subject course_num
bys vccsid: gen audit_tmp = sum(tmp)
bys vccsid strm: egen audit_tmp_2 = max(audit_tmp)
gen audit_tmp_3 = (audit_tmp_2 > 0) // indicator for whether student has ever audited in each term
preserve // for each actively enrolled term, find whether student has ever audited any courses in all prior terms
	keep vccsid strm audit_tmp_3
	duplicates drop
	sort vccsid strm
	rename audit_tmp_3 audit
	reshape wide audit, i(vccsid) j(strm)
	reshape long audit@, i(vccsid) j(strm)
	sort vccsid strm
	bys vccsid: replace audit = audit[_n-1] if _n > 1 & audit == .
	bys vccsid: gen audit_ind = audit[_n-1] if _n > 2
	replace audit_ind = 0 if audit_ind == .
	drop audit
	tempfile audit
	save "`audit'", replace
restore
merge m:1 vccsid strm using "`audit'", keep(3) nogen
*/


* For each student x term, calculate the percentage of withdrawn and incomplete credits in all prior terms enrolled
sort vccsid strm subject course_num
gen withdrawn_tmp = 0
replace withdrawn_tmp = credit if grade == "W"
gen incomplete_tmp = 0
replace incomplete_tmp = credit if inlist(grade, "I", "R")
gen all_tmp = credit
gen dev_tmp = 0
replace dev_tmp = credit if inlist(course_num, "I", "II", "III", "IV") | length(course_num) < 3
foreach v in withdrawn incomplete dev all {
	bys vccsid: gen `v'_sum_tmp = sum(`v'_tmp)
	bys vccsid strm: egen `v'_sum = max(`v'_sum_tmp)
	drop `v'_sum_tmp `v'_tmp
}
gen pct_withdrawn = withdrawn_sum/all_sum
gen pct_incomplete = incomplete_sum/all_sum
gen pct_dev = dev_sum/all_sum

gen passed = inlist(grade, "A", "B", "C", "D", "S", "P", "N", "*", "")
gen prop_tmp = 0
replace prop_tmp = credit if passed == 1
egen term_cred_earn = sum(prop_tmp), by(vccsid strm)
drop passed prop_tmp
gen prop_tmp = 0
replace prop_tmp = credit if !inlist(grade, "W","X","XN")
egen term_cred_att_2 = sum(prop_tmp), by(vccsid strm)
drop prop_tmp
gen prop_comp = term_cred_earn/term_cred_att_2

keep vccsid strm num_of_prior_terms pct_stopped pct_withdrawn pct_incomplete pct_dev prop_comp term_cred_earn term_cred_att_2
duplicates drop
sort vccsid strm
isid vccsid strm
bys vccsid: gen pct_withdrawn_new = pct_withdrawn[_n-1] if _n > 1
bys vccsid: gen pct_incomplete_new = pct_incomplete[_n-1] if _n > 1
bys vccsid: gen pct_dev_new = pct_dev[_n-1] if _n > 1
drop pct_withdrawn pct_incomplete pct_dev
rename pct_withdrawn_new pct_withdrawn
rename pct_incomplete_new pct_incomplete
rename pct_dev_new pct_dev
bys vccsid: gen x = sum(term_cred_earn)
bys vccsid: gen y = sum(term_cred_att_2)
gen overall_prop_comp = x/y
foreach t in 2142 2143 2144 2152 2153 2154 2162 2163 2164 2172 2173 2174 2182 2183 2184 2192 2193 2194 {
	preserve
		keep if strm < `t'
		bys vccsid: egen prop_comp_sd = sd(prop_comp) 
		keep vccsid prop_comp_sd
		duplicates drop
		isid vccsid
		gen strm = `t'
		tempfile prop_`t'
		save "`prop_`t''", replace
	restore
}
preserve
	clear
	foreach t in 2142 2143 2144 2152 2153 2154 2162 2163 2164 2172 2173 2174 2182 2183 2184 2192 2193 2194 {
		append using "`prop_`t''", force
	}
	tempfile prop_sd
	save "`prop_sd'", replace
restore
merge 1:1 vccsid strm using "`prop_sd'", keep(1 3) nogen
sort vccsid strm
bys vccsid: gen overall_prop_comp_new = overall_prop_comp[_n-1] if _n > 1
drop overall_prop_comp
rename overall_prop_comp_new overall_prop_comp
drop x y prop_comp term_cred*

merge 1:1 vccsid strm using "C:\\Users\\ys8mz\\Box Sync\\Clickstream\\data\\bias\\all_vccsid_strm.dta", keep(2 3) nogen
order vccsid strm
sort vccsid strm
replace pct_stopped = 0 if pct_stopped == .
save "C:\\Users\\ys8mz\\Box Sync\\Clickstream\\data\\bias\\four_additional_predictors.dta", replace



***************************************************************
** Create the term GPA and term enrollment intensity predictors
***************************************************************
use "C:\\Users\\ys8mz\\Box Sync\\Clickstream\\data\\bias\\term_lvl_gpa_enrl_intensity.dta", clear
drop term_num
sort vccsid strm
* Enrollment Intensity of the term right before the term in which we'd like to predict course grades
gen enrl_intensity = .
bys vccsid: replace enrl_intensity = term_credits_attempted[_n-1] if _n > 1 & enrl_intensity == .
* Enrollment Intensity of the term in which we'd like to predict course grades
gen crnt_enrl_intensity = term_credits_attempted


* Term GPA for the term right before the term in which we'd like predict course grades (term_gpa_1), as well as the term right before this term (term_gpa_2)
bys vccsid: gen term_num = _n
levelsof strm, local(all_strm)
gen term_gpa_1 = .
gen prev_num = .
foreach s in `all_strm' { // loop through each student x term, and find the term_gpa_1 with respect to the "current" term
	if mod(`s',10) == 2 {
		local prev_s = `s'-8
	}
	else {
		local prev_s = `s'-1
	}
	egen tmp = max(strm) if strm <= `prev_s', by(vccsid)
	egen new_tmp = max(tmp), by(vccsid)
	gen tmp2 = term_gpa if strm == new_tmp
	egen new_tmp2 = max(tmp2), by(vccsid)
	gen tmp3 = term_num if strm == new_tmp
	egen new_tmp3 = max(tmp3), by(vccsid)
	replace term_gpa_1 = new_tmp2 if strm == `s'
	replace prev_num = new_tmp3 if strm == `s'
	drop *tmp *tmp2 *tmp3
}
gen term_gpa_2 = .
foreach s in `all_strm' { // loop through each student x term, and find the term_gpa_2 with respect to the "current" term
	gen tmp = prev_num-1 if strm == `s'
	egen new_tmp = max(tmp), by(vccsid)
	gen tmp2 = term_gpa if new_tmp == term_num
	egen new_tmp2 = max(tmp2), by(vccsid)
	replace term_gpa_2 = new_tmp2 if strm == `s'
	drop *tmp *tmp2
}
drop term_credits_attempted term_gpa
foreach v in enrl_intensity term_gpa_1 term_gpa_2 { // create indicators for availability of enrollment intensity and term GPA predictors
	gen has_`v' = !mi(`v')
	replace `v' = 0 if `v' == .
}


* Merge the different intermediate files
drop term_num prev_num
order vccsid strm crnt_enrl_intensity has_enrl_intensity enrl_intensity has_term_gpa_1 term_gpa_1 has_term_gpa_2 term_gpa_2
sort vccsid strm
merge 1:1 vccsid strm using "C:\\Users\\ys8mz\\Box Sync\\Clickstream\\data\\bias\\all_vccsid_strm.dta", keep(2 3) nogen
sort vccsid strm
tab has_enrl_intensity
drop has_enrl_intensity
save "C:\\Users\\ys8mz\\Box Sync\\Clickstream\\data\\bias\\twelve_additional_predictors.dta", replace // In fact there're a total of 18 predictors stored in this file
