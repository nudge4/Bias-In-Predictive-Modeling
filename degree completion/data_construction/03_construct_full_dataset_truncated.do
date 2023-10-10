/* This script processes all student level VCCS admin data to construct the data set which includes most of the predictors used for degree completion modeling. */

global username="`c(username)'"
global root "/Users/${username}/Box Sync/VCCS data partnership"
* global data "/Users/${username}/Box Sync/VCCS restricted student data"
global data "D:/Yifeng -- Project Work/ys8mz_sandbox"
global main_dir "/Users/ys8mz/Box Sync/Predictive Models of College Completion (VCCS)"



*******************************************************
* Initial processing of term-level academic information
*******************************************************
use "${main_dir}/dta/new/student_level_sample_and_outcomes.dta", clear
* drop if first_nonde_strm > 2123
* merge 1:m vccsid using "${root}/vccs_project_data/intermediate_files/ys8mz/Merged_Class.dta", keep(1 3) nogen
merge 1:m vccsid using "D:\Yifeng -- Project Work\ys8mz_sandbox\Merged_Class.dta", keep(1 3) nogen
keep vccsid strm
duplicates drop
preserve
	* use "${root}/vccs_project_data/intermediate_files/ys8mz/Merged_GPA.dta", clear
	use "D:\Yifeng -- Project Work\ys8mz_sandbox\Merged_GPA.dta", clear
	drop if unt_taken_prgrss == 0
	drop if strm < 2034 | strm > 2184
	tempfile new_gpa
	save "`new_gpa'", replace
restore
merge 1:m vccsid strm using "`new_gpa'"
egen max_merge = max(_merge), by(vccsid)
egen min_merge = min(_merge), by(vccsid)
drop if max_merge == min_merge & max_merge == 2
gen flag_tmp = (_merge == 2)
egen flag = max(flag_tmp), by(vccsid)
drop if flag == 1 // drop the students who have observed enrollment terms according to GPA files but not observed in Class files
keep vccsid
duplicates drop


merge 1:1 vccsid using "${main_dir}/dta/new/student_level_sample_and_outcomes.dta", keep(1 3) nogen
keep vccsid first_nonde_strm grad_vccs_6years grad_nonvccs_6years first_degree_strm
egen grad_6years = rowmax(grad_vccs_6years grad_nonvccs_6years)
gen grad_2years = (first_degree_strm - first_nonde_strm < 20)
gen grad_1years = (first_degree_strm - first_nonde_strm < 10)
* merge 1:m vccsid using "${root}/vccs_project_data/intermediate_files/ys8mz/Merged_Class.dta", keep(1 3) nogen
merge 1:m vccsid using "D:\Yifeng -- Project Work\ys8mz_sandbox\Merged_Class.dta", keep(1 3) nogen
keep vccsid-grad_1years strm
duplicates drop
* merge 1:m vccsid strm using "${root}/vccs_project_data/intermediate_files/ys8mz/Merged_GPA.dta", keep(1 3) nogen
merge 1:m vccsid strm using "D:\Yifeng -- Project Work\ys8mz_sandbox\Merged_GPA.dta", keep(1 3) nogen
replace cur_gpa = . if unt_taken_prgrss == unt_passd_prgrss & unt_taken_prgrss > 0 & unt_taken_prgrss != . & cur_gpa == 0
egen term_cred_att_gpa = sum(unt_taken_prgrss), by(vccsid strm)
replace term_cred_att_gpa = 0 if term_cred_att_gpa == .
drop collnum-unt_taken_prgrss
duplicates drop


gen last_term = first_nonde_strm + 60 // (MODIFY HERE!!) The end of time window should be 3 semesters from the initial enrollment term
replace last_term = last_term - 2 if mod(first_nonde_strm,10) == 4
replace last_term = last_term - 9 if mod(first_nonde_strm,10) != 4
rename last_term old_term
merge m:1 vccsid using "${main_dir}/intermediate_files/new/truncation_nth_df.dta", keep(2 3) nogen
* merge m:1 vccsid using "${root}/vccs_project_data/intermediate_files/ys8mz/agg_cum_gpa_by_term.dta", keep(1 3) nogen
merge m:1 vccsid using "D:\Yifeng -- Project Work\ys8mz_sandbox\agg_cum_gpa_by_term.dta", keep(1 3) nogen
sort vccsid strm
qui levelsof(last_term), local(all_terms)
gen cum_gpa = . // Obviously, we need to use whatever cum_gpa at the truncation points, rather than the actual old_last_term
foreach t in `all_terms' {
	replace cum_gpa = term_`t' if last_term == `t'
}
drop term_2*
drop if strm > last_term
gen earliest_term = first_nonde_strm - 30
* merge 1:m vccsid strm using "${root}/vccs_project_data/intermediate_files/ys8mz/Merged_Class.dta", keep(1 3) nogen
merge 1:m vccsid strm using "D:\Yifeng -- Project Work\ys8mz_sandbox\Merged_Class.dta", keep(1 3) nogen
replace course_num = substr(course_num, 2, .) if regexm(course_num,"^[0-9][0-9][0-9][0-9]L$")
replace course_num = substr(course_num, 1, 3) if regexm(course_num,"^[0-9][0-9][0-9]L$")
sort vccsid college strm subject course_num
gen tmp = 1 if grade == ""
egen flag = max(tmp), by(vccsid college strm subject course_num)
egen num_of_grades_tmp = nvals(grade), by(vccsid college strm subject course_num) 
egen num_of_grades = max(num_of_grades_tmp), by(vccsid college strm subject course_num)
drop num_of_grades_tmp
bys vccsid college strm subject course_num: gen num_of_obs = _N
egen new_grade = mode(grade) if flag == 1 & num_of_grades == 1, by(vccsid college strm subject course_num)
replace grade = new_grade if tmp == 1 & new_grade != ""
drop new_grade tmp flag num_of_grades
drop if credit == 0 & num_of_obs == 1 & mi(grade)
drop num_of_obs
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


replace credit = credit/2 if inlist(grade, "X", "XN")
gen effective_credit_2 = credit
replace effective_credit_2 = 0 if !inlist(grade, "A", "B", "C", "D", "F", "I")
gen numeric_grade = 0
replace numeric_grade = 4 if inlist(grade, "A")
replace numeric_grade = 3 if grade == "B"
replace numeric_grade = 2 if grade == "C"
replace numeric_grade = 1 if grade == "D"
egen term_cred_att_class = sum(credit), by(vccsid strm)
drop if term_cred_att_class == 0 & term_cred_att_gpa == 0
gen flag = (term_cred_att_class < term_cred_att_gpa)
replace credit = credit * term_cred_att_gpa/term_cred_att_class if flag == 1
replace term_cred_att_class = term_cred_att_gpa if flag == 1
drop term_cred_att_gpa
rename term_cred_att_class term_cred_att
drop flag
egen college_cred_att = sum(credit), by(vccsid college)
encode college, gen(college_factor)
drop college
preserve
	keep vccsid college_factor college_cred_att
	duplicates drop
	reshape wide college_cred_att, i(vccsid) j(college_factor)
	sort vccsid
	tempfile entropy
	save "`entropy'", replace
restore
merge m:1 vccsid using "`entropy'", nogen
forvalues i=1/23 {
	replace college_cred_att`i' = 0 if mi(college_cred_att`i')
}
egen total_cred_att = rowtotal(college_cred_att1-college_cred_att23)
forvalues i=1/23 {
	replace college_cred_att`i' = college_cred_att`i'/total_cred_att
}
egen max_cred_att = rowmax(college_cred_att1-college_cred_att23)
gen college_entropy = -log(max_cred_att)
drop college_cred_att-max_cred_att
gen tmp = credit if grade == "W"
replace tmp = 0 if grade != "W"
egen term_withdrawn = sum(tmp), by(vccsid strm)
drop tmp
gen passed = inlist(grade, "A", "B", "C", "D", "S", "P", "N", "*", "")
gen tmp = 0
replace tmp = credit if passed == 1
egen term_cred_earn = sum(tmp), by(vccsid strm)
drop passed tmp
gen tmp = 0
replace tmp = credit if !inlist(grade, "W","X","XN")
egen term_cred_att_2 = sum(tmp), by(vccsid strm)
drop tmp
gen tmp = 0
replace tmp = credit if regexm(course_num, "^2[0-9][0-9]$") & !inlist(grade, "W","X","XN")
egen term_lvl2_att = sum(tmp), by(vccsid strm)
drop tmp
gen tmp = 0
replace tmp = credit if !regexm(course_num, "^[12][0-9][0-9]$") & !inlist(grade, "W","X","XN")
egen term_dev_att = sum(tmp), by(vccsid strm)
drop tmp
egen sum_effective_credit_2 = sum(effective_credit_2), by(vccsid strm)
gen grade_points = effective_credit_2*numeric_grade
egen sum_grade_points = sum(grade_points), by(vccsid strm)
gen term_gpa = sum_grade_points/sum_effective_credit_2
* drop sum_* effective_credit_2 grade_points numeric_grade acadplan-acad_group
drop sum_* effective_credit_2 grade_points numeric_grade acad_group-year
duplicates drop
sort vccsid strm
drop if strm < first_nonde_strm
save "${main_dir}/intermediate_files/new/temp_1.dta", replace



********************************************************************************************
** Load and process the Financial files one by one, and append it to the running merged file
********************************************************************************************
clear
local files : dir "${data}/Raw/FinancialAid" files "*.csv"
foreach file in `files' {
		di "`file'"
		local yr = substr("`file'",2,4)
		preserve
			import delimited using "${data}/Raw/FinancialAid/`file'", clear stringcols(_all)
			duplicates drop
			foreach v in acg ctg ctgplus gearup grsef grsnbef loanef locgov othef othin othoth outinst priloan smart stemef wsot {
				replace `v' = "0" if `v' == "NA"
			}
			gen year = `yr'
			tempfile finaid_file
			save "`finaid_file'", replace
		restore
		append using "`finaid_file'", force
}
* drop if year > 2013 // (MODIFY HERE!!)
drop aidwin athperc calsys credwin exce gender grdcom lastdol level locdomi part race rectype repper repyear stustat tagu tuition vaguap visa vocreh vsp vtg zip // keep "budget" column here!
replace budget = "000000" if budget == "XXXXXX" & pell != "0"
replace budget = "" if budget == "XXXXXX"
destring acg-wsot, replace
order vccsid year fice aidsum aidfal aidspr budget credfal credspr credsum
egen tot_aids = rowtotal(acg-wsot)
gen aid_avail = (aidfal+aidspr+aidsum > 0)
assert tot_aids == 0 if aid_avail == 0
* drop if tot_aids == 0
sort vccsid year fice
duplicates drop
collapse (max) aidsum aidfal aidspr budget credsum credfal credspr (sum) acg-wsot, by(vccsid year fice)
gen credall = credsum+credfal+credspr
foreach v in "sum" "fal" "spr" {
	replace cred`v' = 1 if credall == 0 & aid`v' == 1
}
order vccsid-credspr credall
foreach v in "sum" "fal" "spr" {
	replace cred`v' = 0 if aid`v' == 0
}
replace credall = credsum + credfal + credspr
foreach v in "sum" "fal" "spr" {
	replace cred`v' = 1 if credall == 0 & aid`v' == 1
}
replace credall = credsum + credfal + credspr
foreach v in "sum" "fal" "spr" {
	gen pct`v' = cred`v'/credall
}
drop aidsum-aidspr 
drop credsum-credall
order vccsid year fice budget pctsum pctfal pctspr
sort vccsid year fice
gen tot_grants = acg + csap + ctg + ctgplus + discaid + gearup + grantin + grsef + grsnbef + hetap + locgov + msdawd + msdtfw + othef + othfed + othin + othoth + outinst + pell + ptap + schoin + seog + smart + tag + tuiwaiv + tviigrants + vgap
gen tot_sub_loans = perkins + staloa
gen tot_unsub_loans = loanef + loanin + plusloa + priloan + staloun + tviiloans
gen tot_others = cwsp + stemef + stemin + wsot
keep vccsid-pctspr pell tot*
rename pell tot_pell
foreach v in pell grants sub_loans unsub_loans others {
	foreach t in sum fal spr {
		gen `v'_`t' = tot_`v' * pct`t'
	}
}
drop pct* tot_*
foreach v in pell grants sub_loans unsub_loans others {
	foreach t in sum fal spr {
		if "`t'" == "sum" {
			rename `v'_`t' `v'3
		}
		else if "`t'" == "fal" {
			rename `v'_`t' `v'4
		}
		else {
			rename `v'_`t' `v'2
		}
	}
}
replace budget = 999999 if budget == .
reshape long pell@ grants@ sub_loans@ unsub_loans@ others@, i(vccsid year fice budget) j(term)
replace year = year+1 if term == 2
replace pell = . if pell == 0 & budget == 999999
sort vccsid year term
collapse (sum) pell grants sub_loans unsub_loans others, by(vccsid year term budget)
replace pell = . if pell == 0 & budget == 999999
replace pell = (pell > 0) if pell != .
collapse (max) pell (sum) grants sub_loans unsub_loans others, by(vccsid year term)
tostring year, replace
tostring term, replace
gen strm = substr(year,1,1) + substr(year,3,2) + term
drop year term
order vccsid strm
destring strm, replace
sort vccsid strm
* drop if grants+sub_loans+unsub_loans+others == 0
save "${main_dir}/intermediate_files/new/temp_2.dta", replace
use "${main_dir}/intermediate_files/new/temp_1.dta", clear
keep vccsid first_nonde_strm last_term
duplicates drop
sort vccsid
merge 1:m vccsid using "${main_dir}/intermediate_files/new/temp_2.dta", keep(3) nogen
drop if strm < first_nonde_strm | strm > last_term
sort vccsid strm
gen flag = int((strm - first_nonde_strm) / 10) + 1
// replace flag = 2 if strm - first_nonde_strm >= 10 //(MODIFY HERE!!)
foreach v in grants sub_loans unsub_loans others {
	foreach t in 1 2 3 4 5 6 { // (MODIFY HERE!!)
		egen `v'_yr`t'_tmp = sum(`v') if flag == `t', by(vccsid)
		egen `v'_yr`t' = max(`v'_yr`t'_tmp), by(vccsid)
		drop `v'_yr`t'_tmp
		replace `v'_yr`t' = 0 if `v'_yr`t' == .
	}
	drop `v'
}
drop flag
drop first_nonde_strm last_term
sort vccsid strm
save "${main_dir}/intermediate_files/new/temp_3.dta", replace


use "${main_dir}/intermediate_files/new/temp_1.dta", clear
keep vccsid first_nonde_strm last_term strm
sort vccsid strm
gen enrolled = 1
reshape wide enrolled, i(vccsid first_nonde_strm last_term) j(strm)
reshape long enrolled@, i(vccsid first_nonde_strm last_term) j(strm)
drop if strm < first_nonde_strm | strm > last_term
replace enrolled = 0 if enrolled == .
label drop college_factor
sort vccsid strm
merge 1:1 vccsid strm using "${main_dir}/intermediate_files/new/temp_3.dta", keep(1 3) nogen
replace pell = . if enrolled == 0 // This is a subjective decision (3/9/2020): we don't look at the pell eligibility information in non-enrolled terms
foreach v in grants sub_loans unsub_loans others {
	foreach t in yr1 yr2 yr3 yr4 yr5 yr6 { //(MODIFY HERE!!)
		replace `v'_`t' = 0 if `v'_`t' == .
		egen `v'_`t'_new = max(`v'_`t'), by(vccsid)
		drop `v'_`t'
		rename `v'_`t'_new `v'_`t'
	}
}
egen pell_pct = max(pell), by(vccsid)
sort vccsid strm
save "${main_dir}/intermediate_files/new/temp_4.dta", replace



**********************************
* Process degree/major information
**********************************
clear
** Load the Build Student files one by one, and append it to the running merged file
local files : dir "${data}/Build/Student" files "*.dta.zip"
foreach file in `files' {
	di "`file'"
	preserve
		zipuse "${data}/Build/Student/`file'", clear
		keep vccsid strm college curr curr_degree total_credit_hrs dual_enrollment hs_grad_year
		gen degree_lvl = 0
		replace degree_lvl = 1 if curr_degree == "DIPL"
		replace degree_lvl = 2 if curr_degree == "CSC"
		replace degree_lvl = 3 if curr_degree == "CERT"
		replace degree_lvl = 4 if regexm(curr_degree, "A")
		drop curr_degree
		sort vccsid strm college
		duplicates drop
		tempfile curr_data
		save "`curr_data'", replace
	restore
	append using "`curr_data'", force
}
drop if strm > 2183 // (MODIFY HERE!!) discard the observations which certainly come after the end of the 17th term
egen dual = max(dual_enrollment), by(vccsid strm)
gsort vccsid strm -total_credit_hrs -degree_lvl
set seed 1234
gen rn = runiform()
gsort vccsid strm -total_credit_hrs -degree_lvl rn
bys vccsid strm: keep if _n == 1
gen degree_seeking = !regexm(curr, "^0")
drop dual_enrollment degree_lvl rn total_credit_hrs college
sort vccsid strm
save "${main_dir}/intermediate_files/new/temp_5.dta", replace


use "${main_dir}/intermediate_files/new/temp_4.dta", clear // temp_4.dta contains all terms during the non-DE enrollment window (all six years since initial enrollment term) for each student
merge 1:1 vccsid strm using "${main_dir}/intermediate_files/new/temp_5.dta"
replace degree_seeking = 1 if enrolled == 1 & degree_seeking == .
replace degree_seeking = 0 if degree_seeking == . | enrolled == 0
replace curr = "" if enrolled == 0
sort vccsid strm
foreach v in hs_grad_year {
	egen `v'_new = mode(`v'), maxmode by(vccsid)
	drop `v'
	rename `v'_new `v'	
}
egen dual_ind = max(dual), by(vccsid)
drop if _merge == 2
drop dual _merge
gen first_nonde_yr = floor(first_nonde_strm/10) + 1800
gen seamless_enrollee_0 = (hs_grad_year < first_nonde_yr)
gen seamless_enrollee_1 = (hs_grad_year == first_nonde_yr)
drop hs_grad_year first_nonde_yr
egen num_of_curr_tmp = nvals(curr), by(vccsid)
egen num_of_curr = max(num_of_curr_tmp), by(vccsid)
replace num_of_curr = 0 if num_of_curr == .
gen program_chng_ind = (num_of_curr > 1)
drop curr num_of_curr_tmp num_of_curr
save "${main_dir}/intermediate_files/new/temp_6.dta", replace


***********************************************************************************
** Identify credits attempted/earned during pre period based on GPA and Class files
***********************************************************************************
use "${main_dir}/intermediate_files/new/temp_6.dta", clear
keep vccsid first_nonde_strm
duplicates drop
* merge 1:m vccsid using "${root}/vccs_project_data/intermediate_files/ys8mz/Merged_GPA.dta", keep(1 3) nogen
merge 1:m vccsid using "D:\Yifeng -- Project Work\ys8mz_sandbox\Merged_GPA.dta", keep(1 3) nogen
keep if first_nonde_strm > strm
sort vccsid strm collnum
drop cum_gpa cur_gpa institution unt_*
sort vccsid collnum strm
bys vccsid collnum: keep if _n == _N
reshape wide tot_taken_prgrss tot_passd_prgrss, i(vccsid first_nonde_strm collnum) j(strm)
reshape long tot_taken_prgrss@ tot_passd_prgrss@, i(vccsid first_nonde_strm collnum) j(strm)
label drop _all
drop if strm >= first_nonde_strm
sort vccsid collnum strm
bys vccsid collnum: replace tot_taken_prgrss = tot_taken_prgrss[_n-1] if tot_taken_prgrss[_n-1] != . & tot_taken_prgrss == .
bys vccsid collnum: replace tot_passd_prgrss = tot_passd_prgrss[_n-1] if tot_passd_prgrss[_n-1] != . & tot_passd_prgrss == .
bys vccsid collnum: keep if _n == _N
drop if tot_taken_prgrss == 0
foreach v in tot_taken_prgrss tot_passd_prgrss {
	egen `v'_new = sum(`v'), by(vccsid)
	drop `v'
	rename `v'_new `v'
} // Identify the aggregated total credits taken/passed prior to initial enrollment term, if the student attends multiple VCCS schools during the pre-window (three years)
drop collnum first_nonde_strm strm
duplicates drop
gen enrolled_pre = 1
preserve
	use "${main_dir}/intermediate_files/new/temp_6.dta", clear
	keep vccsid first_nonde_strm
	duplicates drop
	sort vccsid
	tempfile all_sample
	save "`all_sample'", replace
restore
merge 1:1 vccsid using "`all_sample'", nogen
preserve
	* merge 1:m vccsid using "${root}/vccs_project_data/intermediate_files/ys8mz/Merged_Class.dta", keep(1 3) nogen
	merge 1:m vccsid using "D:\Yifeng -- Project Work\ys8mz_sandbox\Merged_Class.dta", keep(1 3) nogen
	drop if strm >= first_nonde_strm
	drop if credit == 0
	replace course_num = substr(course_num, 2, .) if regexm(course_num,"^[0-9][0-9][0-9][0-9]L$")
	replace course_num = substr(course_num, 1, 3) if regexm(course_num,"^[0-9][0-9][0-9]L$")
	gen enrolled_pre_new = 1
	egen prev_cred_att_tmp = sum(credit) if !inlist(grade, "W", "X"), by(vccsid)
	egen prev_cred_att = max(prev_cred_att_tmp), by(vccsid)
	drop prev_cred_att_tmp
	egen prev_cred_earn_tmp = sum(credit) if inlist(grade, "A", "B", "C", "D", "P", "S", "N", "*", ""), by(vccsid)
	egen prev_cred_earn = max(prev_cred_earn_tmp), by(vccsid)
	drop prev_cred_earn_tmp
	egen tmp = sum(credit) if inlist(grade, "A", "B", "C", "D") & !regexm(course_num, "^[12][0-9][0-9]$"), by(vccsid)
	egen prev_dev_earn = max(tmp), by(vccsid)
	drop tmp
	foreach v in cred_att cred_earn dev_earn {
		replace prev_`v' = 0 if prev_`v' == .
	}
	keep vccsid-first_nonde_strm enrolled_pre_new-prev_dev_earn
	duplicates drop
	drop enrolled_pre first_nonde_strm
	sort vccsid
	tempfile tmp1
	save "`tmp1'", replace
restore
merge 1:1 vccsid using "`tmp1'", keep(1 3) nogen
replace enrolled_pre_new = 1 if enrolled_pre == 1 & enrolled_pre_new == .
drop enrolled_pre
drop if enrolled_pre_new == .
sort vccsid
gen flag = mi(tot_taken_prgrss)
replace tot_taken_prgrss = prev_cred_att if flag == 1
replace tot_passd_prgrss = prev_cred_earn if flag == 1
rename enrolled_pre_new enrolled_pre
replace prev_dev_earn = 0 if mi(prev_dev_earn)
save "${main_dir}/intermediate_files/new/temp_7.dta", replace


***********************************************************************************
** Identify credits attempted/earned during pre period based on Student files,
** and finalize the predictors of coll_lvl_cred_earn, prop_comp_pre and cum_gpa_pre
***********************************************************************************
clear
** Load the Build Student files one by one, and append it to the running merged file
local files : dir "${data}/Build/Student" files "*.dta.zip"
foreach file in `files' {
	di "`file'"
	preserve
		zipuse "${data}/Build/Student/`file'", clear
		keep vccsid strm college curr curr_degree total_credit_hrs
		gen degree_lvl = 0
		replace degree_lvl = 1 if curr_degree == "DIPL"
		replace degree_lvl = 2 if curr_degree == "CSC"
		replace degree_lvl = 3 if curr_degree == "CERT"
		replace degree_lvl = 4 if regexm(curr_degree, "A")
		drop curr_degree
		sort vccsid strm college
		duplicates drop
		tempfile curr_data
		save "`curr_data'", replace
	restore
	append using "`curr_data'", force
}
* drop if strm > 2132 // (MODIFY HERE!!)
drop if total_credit_hrs == 0
drop college curr degree_lvl
collapse (sum) total_credit_hrs, by(vccsid strm)
sort vccsid strm
preserve
	use "${main_dir}/intermediate_files/new/temp_6.dta", clear
	keep vccsid first_nonde_strm
	duplicates drop
	sort vccsid
	tempfile all_sample
	save "`all_sample'", replace
restore
merge m:1 vccsid using "`all_sample'", keep(2 3) nogen
keep if strm < first_nonde_strm
drop strm
collapse (sum) total_credit_hrs, by(vccsid)
gen enrolled_pre_new = 1
sort vccsid
merge 1:1 vccsid using "${main_dir}/intermediate_files/new/temp_7.dta", nogen
replace enrolled_pre = 1 if enrolled_pre == . & enrolled_pre_new == 1
gen coll_lvl_cred_earn = total_credit_hrs if flag == .
replace coll_lvl_cred_earn = total_credit_hrs - prev_dev_earn if flag == 1 & total_credit_hrs ! = .
replace coll_lvl_cred_earn = tot_passd_prgrss - prev_dev_earn if coll_lvl_cred_earn == .
keep vccsid enrolled_pre tot_taken_prgrss tot_passd_prgrss coll_lvl_cred_earn
order vccsid enrolled_pre tot_taken_prgrss tot_passd_prgrss coll_lvl_cred_earn
sort vccsid
gen prop_comp_pre = tot_passd_prgrss/tot_taken_prgrss
drop tot_*
preserve
	use "${main_dir}/intermediate_files/new/temp_6.dta", clear
	keep vccsid first_nonde_strm
	duplicates drop
	sort vccsid
	tempfile all_sample
	save "`all_sample'", replace
restore
merge 1:1 vccsid using "`all_sample'", keep(3) nogen
* merge 1:1 vccsid using "${root}/vccs_project_data/intermediate_files/ys8mz/agg_cum_gpa_by_term.dta", keep(1 3) nogen
merge 1:1 vccsid using "D:\Yifeng -- Project Work\ys8mz_sandbox\agg_cum_gpa_by_term.dta", keep(1 3) nogen
drop term_2133-term_2213
reshape long term_@, i(vccsid enrolled_pre coll_lvl_cred_earn prop_comp_pre first_nonde_strm) j(strm)
drop if strm > first_nonde_strm
rename term_ cum_gpa_pre
label drop _all
sort vccsid strm
bys vccsid: keep if _n == _N
drop strm
gen flag = mi(cum_gpa_pre)
merge 1:m vccsid using "D:\Yifeng -- Project Work\ys8mz_sandbox\Merged_Class.dta", keep(1 3) nogen
* merge 1:m vccsid using "${root}/vccs_project_data/intermediate_files/ys8mz/Merged_Class.dta", keep(1 3) nogen
gen tmp = (strm < first_nonde_strm)
replace course_num = substr(course_num, 2, .) if regexm(course_num,"^[0-9][0-9][0-9][0-9]L$")
replace course_num = substr(course_num, 1, 3) if regexm(course_num,"^[0-9][0-9][0-9]L$")
egen sum_credit_tmp = sum(credit) if flag == 1 & tmp == 1 & inlist(grade, "A", "B", "C", "D", "F", "I"), by(vccsid)
gen numeric_grade = 0
replace numeric_grade = 4 if grade == "A"
replace numeric_grade = 3 if grade == "B"
replace numeric_grade = 2 if grade == "C"
replace numeric_grade = 1 if grade == "D"
gen grade_points = numeric_grade*credit
egen sum_grade_points_tmp = sum(grade_points) if flag == 1 & tmp == 1 & inlist(grade, "A", "B", "C", "D", "F", "I"), by(vccsid)
foreach v in grade_points credit {
	egen sum_`v' = max(sum_`v'_tmp), by(vccsid)
}
gen cum_gpa_pre_new = sum_grade_points/sum_credit
replace cum_gpa_pre = cum_gpa_pre_new if !mi(cum_gpa_pre_new)
keep vccsid-cum_gpa_pre
drop first_nonde_strm
sort vccsid
replace coll_lvl_cred_earn = coll_lvl_cred_earn/30 // divide the raw college-level credits earned by 30 to normalize this predictor
duplicates drop
save "${main_dir}/intermediate_files/new/temp_8.dta", replace



************************************
* Process NSC enrollment information
************************************
clear
forvalues y=2004/2018 { // (MODIFY HERE!!)
	di "`y'"
	preserve
		zipuse "${data}/Build/NSC/NSC_enrollment_`y'.dta.zip", clear
		drop if state=="VA" & two_four==2 & type=="Public" & college!="Richard Bland College"
		keep vccsid college enrol_begin enrol_end state two_four type enrol_status ipeds
		tempfile nsc_year
		save "`nsc_year'", replace
	restore
	append using "`nsc_year'", force
}
gen instate = (state == "VA")
label drop two_four
gen four_year = (two_four == 4)
drop two_four
gen public = (type == "Public")
drop type
preserve
	keep if mi(ipeds)
	keep college state ipeds
	sort college state
	duplicates drop
	replace ipeds = 999999
	tostring ipeds, replace
	gen part2 = _n
	tostring part2, replace
	replace ipeds = ipeds + "-" + part2
	drop part2
	rename ipeds new_ipeds
	tempfile new_cc
	save "`new_cc'", replace
restore
merge m:1 college state using "`new_cc'", keep(1 3) nogen
tostring ipeds, replace
replace new_ipeds = ipeds if mi(new_ipeds)
assert !mi(new_ipeds)
drop ipeds
rename new_ipeds college_code
drop college state
sort vccsid enrol_begin enrol_end
rename college_code college
duplicates drop
decode enrol_status, gen(enrol_status_str)
gen enrol_status_new = .
replace enrol_status_new = 1 if enrol_status_str == "Full-Time"
replace enrol_status_new = 0.5 if enrol_status_str == "Half-Time"
replace enrol_status_new = 0.75 if enrol_status_str == "Three-Quarter Time"
replace enrol_status_new = 0.25 if enrol_status_str == "Less than Half-Time"
replace enrol_status_new = 0 if inlist(enrol_status_str, "Leave of Absence", "Withdrew")
drop enrol_status enrol_status_str
rename enrol_status_new nsc_enrl_intensity
preserve
	keep college four_year public instate
	order college four_year public instate
	sort college
	duplicates drop
	foreach v in four_year public instate {
		egen new_`v' = max(`v'), by(college)
		drop `v'
		rename new_`v' `v'
	}
	duplicates drop
	isid college
	sort college
	tostring four_year public instate, replace
	gen college_type = four_year+public+instate
	drop four_year public instate
	encode college_type, gen(college_type_new)
	drop college_type
	label drop college_type_new
	gen nsc_coll_type_ = 1
	reshape wide nsc_coll_type_, i(college) j(college_type_new)
	forvalues i=1/8 {
		replace nsc_coll_type_`i' = 0 if nsc_coll_type_`i' == .
	}
	save "${main_dir}/intermediate_files/new/temp_nsc_coll_type.dta", replace
restore
preserve
	use "${main_dir}/intermediate_files/new/temp_1.dta", clear
	keep vccsid first_nonde_strm last_term earliest_term
	duplicates drop
	sort vccsid
	tempfile all_sample
	save "`all_sample'", replace
restore
merge m:1 vccsid using "`all_sample'", keep(3) nogen
foreach v in begin end {
	gen `v'_year = year(enrol_`v')
	gen `v'_month = month(enrol_`v')
}
forvalues y=204/218 { // (MODIFY HERE!!)
	local yr = `y'+1800
	foreach t in 2 3 4 {
		if `t' == 2 {
			gen enrolled_`y'`t' = 0
			replace enrolled_`y'`t' = 1 if ((begin_year == `yr' & begin_month <= 4) | begin_year < `yr') & ((end_year == `yr' & end_month >= 1) | end_year > `yr')
		}
		else if `t' == 3 {
			gen enrolled_`y'`t' = 0
			replace enrolled_`y'`t' = 1 if ((begin_year == `yr' & begin_month <= 7) | begin_year < `yr') & ((end_year == `yr' & end_month >= 6) | end_year > `yr')
		}
		else {
			gen enrolled_`y'`t' = 0
			replace enrolled_`y'`t' = 1 if ((begin_year == `yr' & begin_month <= 12) | begin_year < `yr') & ((end_year == `yr' & end_month >= 9) | end_year > `yr')
		}
		gen enrolled_intensity_`y'`t' = 0
		replace enrolled_intensity_`y'`t' = nsc_enrl_intensity if enrolled_`y'`t' == 1
	}
}
sort vccsid college
forvalues y=204/218 { // (MODIFY HERE!!)
	foreach t in 2 3 4 {
		egen enrolled_`y'`t'_new = max(enrolled_`y'`t'), by(vccsid college)
		drop enrolled_`y'`t'
		rename enrolled_`y'`t'_new enrolled_`y'`t'
		replace enrolled_intensity_`y'`t' = 0.01 if enrolled_intensity_`y'`t' == .
		egen enrolled_intensity_`y'`t'_new = max(enrolled_intensity_`y'`t'), by(vccsid college)
		drop enrolled_intensity_`y'`t'
		rename enrolled_intensity_`y'`t'_new enrolled_intensity_`y'`t'
		replace enrolled_intensity_`y'`t' = . if round(enrolled_intensity_`y'`t',0.01) == 0.01
	}
}
drop enrolled_2042 enrolled_intensity_2042 instate four_year public nsc_enrl_intensity enrol_* begin_* end_* // (MODIFY HERE!!)
duplicates drop
sort vccsid college
reshape long enrolled_@ enrolled_intensity_@, i(vccsid college first_nonde_strm last_term earliest_term) j(strm)
label drop _all
rename enrolled_ enrolled_nsc
rename enrolled_intensity enrl_intensity_nsc
drop if strm < earliest_term | strm > last_term
gen pre_flag = (strm < first_nonde_strm)
save "${main_dir}/intermediate_files/new/temp_nsc_enrl.dta", replace
drop enrl_intensity_nsc
egen tmp2 = nvals(college) if pre_flag == 1 & enrolled_nsc == 1, by(vccsid)
egen pre_num_nsc_coll = max(tmp2), by(vccsid)
egen tmp4 = nvals(college) if pre_flag == 0 & enrolled_nsc == 1, by(vccsid)
egen num_nsc_coll = max(tmp4), by(vccsid)
drop tmp2 tmp4
replace pre_num_nsc_coll = 0 if pre_num_nsc_coll == .
replace num_nsc_coll = 0 if num_nsc_coll == .
collapse (max) enrolled_nsc, by(vccsid first_nonde_strm last_term earliest_term strm pre_flag pre_num_nsc_coll num_nsc_coll)
egen tmp = sum(enrolled_nsc) if pre_flag == 1, by(vccsid)
egen pre_nsc_terms = max(tmp), by(vccsid)
egen tmp3 = sum(enrolled_nsc) if pre_flag == 0, by(vccsid)
egen nsc_terms = max(tmp3), by(vccsid)
drop tmp tmp3 first_nonde_strm-pre_flag enrolled_nsc
duplicates drop
sort vccsid
drop if pre_nsc_terms == 0 & nsc_terms == 0
save "${main_dir}/intermediate_files/new/temp_9.dta", replace



********************************************
* Incorporate NSC school quality information
********************************************
insheet using "/Users/${username}/Box Sync/NSC_College_Quality/Raw data/Most-Recent-Cohorts-All-Data-Elements.csv", clear // Download "Most recent data" from https://collegescorecard.ed.gov/data/
keep UNITID adm_rate* satvr* satmt* satwr* c150_4 c150_l4 //choosing college-level variables of interest to describe quality of institution
drop *mid
rename UNITID college_code
tostring college_code, replace
foreach var of varlist _all {
	if "`var'" != "college_code" {
		capture replace `var'="" if `var'=="NULL"
		capture destring `var', replace
	}
}
replace adm_rate = adm_rate_all if mi(adm_rate)
replace c150_4 = c150_l4 if mi(c150_4)
drop adm_rate_all c150_l4
rename adm_rate admrate
rename c150_4 gradrate
sort college_code
rename college_code college
save "${main_dir}/intermediate_files/new/temp_nsc_quality.dta", replace



********************************************
* Create additional academic predictors
********************************************
use "${main_dir}/intermediate_files/new/temp_1.dta", clear
keep vccsid last_term cum_gpa
duplicates drop
sort vccsid
* merge 1:m vccsid using "${root}/vccs_project_data/intermediate_files/ys8mz/Merged_Class.dta", keep(1 3) nogen
merge 1:m vccsid using "D:\Yifeng -- Project Work\ys8mz_sandbox\Merged_Class.dta", keep(1 3) nogen
gen tmp = (strm <= last_term)
gen flag = mi(cum_gpa)
replace course_num = substr(course_num, 2, .) if regexm(course_num,"^[0-9][0-9][0-9][0-9]L$")
replace course_num = substr(course_num, 1, 3) if regexm(course_num,"^[0-9][0-9][0-9]L$")
egen sum_credit_tmp = sum(credit) if flag == 1 & tmp == 1 & inlist(grade, "A", "B", "C", "D", "F", "I"), by(vccsid)
gen numeric_grade = 0
replace numeric_grade = 4 if grade == "A"
replace numeric_grade = 3 if grade == "B"
replace numeric_grade = 2 if grade == "C"
replace numeric_grade = 1 if grade == "D"
gen grade_points = numeric_grade*credit
egen sum_grade_points_tmp = sum(grade_points) if flag == 1 & tmp == 1 & inlist(grade, "A", "B", "C", "D", "F", "I"), by(vccsid)
foreach v in grade_points credit {
	egen sum_`v' = max(sum_`v'_tmp), by(vccsid)
}
gen cum_gpa_new = sum_grade_points/sum_credit
replace cum_gpa = cum_gpa_new if !mi(cum_gpa_new)
keep vccsid cum_gpa
duplicates drop
sort vccsid 
save "${main_dir}/intermediate_files/new/temp_10.dta", replace // new cum_gpa


use "${main_dir}/intermediate_files/new/temp_1.dta", clear
keep vccsid strm
rename strm enrolled_strm
sort vccsid enrolled_strm
bys vccsid: gen obs_id = _n
forvalues i=1/17 { // (MODIFY HERE!!)
	preserve
		keep if obs_id == `i'
		* merge 1:m vccsid using "${root}/vccs_project_data/intermediate_files/ys8mz/Merged_Class.dta", keep(1 3) nogen
		merge 1:m vccsid using "D:\Yifeng -- Project Work\ys8mz_sandbox\Merged_Class.dta", keep(1 3) nogen
		keep if strm <= enrolled_strm
		sort vccsid strm
		tempfile repeat`i'
		save "`repeat`i''", replace
	restore
}
clear
forvalues i=1/17 { // (MODIFY HERE!!)
	append using "`repeat`i''", force
}
drop if inlist(grade,"X", "XN", "W")
sort vccsid enrolled_strm strm
replace course_num = substr(course_num, 2, .) if regexm(course_num,"^[0-9][0-9][0-9][0-9]L$")
replace course_num = substr(course_num, 1, 3) if regexm(course_num,"^[0-9][0-9][0-9]L$")
gen course = subject + course_num
gen crnt_ind = (strm == enrolled_strm)
keep vccsid enrolled_strm crnt_ind course
duplicates drop
sort vccsid enrolled_strm crnt_ind course
preserve
	use "/Users/${username}/Box Sync/VCCS public data/VCCS_Course_List/VCCS_repeatable_courses.dta", clear
	rename course_id course
	duplicates drop
	sort course
	tempfile repeatable
	save "`repeatable'", replace
restore
merge m:1 course using "`repeatable'", keep(1) nogen
drop repeatable
sort vccsid enrolled_strm crnt_ind course
drop crnt_ind
sort vccsid enrolled_strm course
bys vccsid enrolled_strm course: gen obs_id = _n
bys vccsid enrolled_strm course: keep if _n == _N
rename obs_id repeated
replace repeated = repeated-1
collapse (sum) repeated, by(vccsid enrolled_strm)
rename enrolled_strm strm
sort vccsid strm
save "${main_dir}/intermediate_files/new/temp_11.dta", replace



**************************
* Create the age predictor
**************************
clear
** Load the Build Student files one by one, and append it to the running merged file
local files : dir "${data}/Build/Student" files "*.dta.zip"
foreach file in `files' {
	di "`file'"
	preserve
		zipuse "${data}/Build/Student/`file'", clear
		keep vccsid strm age
		egen new_age = mode(age), minmode by(vccsid)
		drop age
		rename new_age age
		duplicates drop
		sort vccsid
		tempfile age
		save "`age'", replace
	restore
	append using "`age'", force
}
drop if strm < 2073 | strm > 2132 // we need to grab age_entry
rename strm first_nonde_strm
preserve
	use "${main_dir}/intermediate_files/new/temp_1.dta", clear
	keep vccsid first_nonde_strm
	duplicates drop
	sort vccsid
	tempfile all_sample
	save "`all_sample'", replace
restore
merge m:1 vccsid first_nonde_strm using "`all_sample'", keep(2 3) nogen
sort vccsid
drop first_nonde_strm
save "${main_dir}/intermediate_files/new/temp_13.dta", replace



*********************************
* Merge everything created so far
*********************************
use "${main_dir}/intermediate_files/new/temp_13.dta", clear
rename age age_entry
* merge 1:1 vccsid using "${root}/vccs_project_data/intermediate_files/ys8mz/merged_student_demographics.dta", keep(1 3) nogen
merge 1:1 vccsid using "D:\Yifeng -- Project Work\ys8mz_sandbox\merged_student_demographics.dta", keep(1 3) nogen
drop age
* merge 1:1 vccsid using "${root}/vccs_project_data/intermediate_files/ys8mz/phe_predictors.dta", keep(1 3) nogen
merge 1:1 vccsid using "D:\Yifeng -- Project Work\ys8mz_sandbox\post_MVP\phe_predictors.dta", keep(1 3) nogen
forvalues i=1/7 {
	gen phe_`i' = (phe == `i')
}
drop has_phe phe
preserve 
	use "${main_dir}/intermediate_files/new/temp_6.dta", clear
	keep vccsid grants_yr1-program_chng_ind
	drop degree_seeking
	duplicates drop
	tempfile part_tmp
	save "`part_tmp'", replace
restore
merge 1:1 vccsid using "`part_tmp'", keep(1 3) nogen
merge 1:1 vccsid using "${main_dir}/intermediate_files/new/temp_8.dta", keep(1 3) nogen
foreach v in prop_comp_pre cum_gpa_pre {
	replace `v' = 0 if enrolled_pre == .
}
foreach v in enrolled_pre coll_lvl_cred_earn {
	replace `v' = 0 if `v' == .
}
merge 1:1 vccsid using "${main_dir}/intermediate_files/new/temp_9.dta", keep(1 3) nogen
foreach v in pre_num_nsc_coll pre_nsc_terms num_nsc_coll nsc_terms {
	replace `v' = 0 if `v' == .
}
rename pell_pct pell_ind
gen pell_0_ind = (pell_ind == 0)
gen pell_1_ind = (pell_ind == 1)
merge 1:1 vccsid using "${main_dir}/intermediate_files/new/temp_10.dta", keep(1 3) nogen
preserve 
	use "${main_dir}/intermediate_files/new/temp_11.dta", clear
	collapse (max) repeated, by(vccsid)
	replace repeated = (repeated > 0)
	sort vccsid
	tempfile repeat_ever
	save "`repeat_ever'", replace
restore
merge 1:1 vccsid using "`repeat_ever'", keep(1 3) nogen
replace repeated = 0 if repeated == .
rename repeated repeat_ind
merge 1:1 vccsid using "${main_dir}/intermediate_files/new/temp_12.dta", keep(3) nogen // (MODIFY HERE!!)
sort vccsid
save "${main_dir}/intermediate_files/new/part_1.dta", replace



*********************************
* Create term-specific predictors
*********************************
use "${main_dir}/intermediate_files/new/temp_6.dta", clear
keep vccsid strm first_nonde_strm last_term enrolled pell degree_seeking
merge 1:1 vccsid strm first_nonde_strm last_term using "${main_dir}/intermediate_files/new/temp_1.dta", keep(1 3) nogen
drop earliest_term cum_gpa old_term truncated nth
foreach v in grad_vccs_6years grad_nonvccs_6years first_degree_strm grad_6years grad_2years grad_1years {
	egen `v'_new = max(`v'), by(vccsid)
	drop `v'
	rename `v'_new `v'
}
order vccsid first_nonde_strm last_term grad_vccs_6years-grad_1years
gen prop_comp = term_cred_earn/term_cred_att_2
gen withdrawn_prop_comp = term_withdrawn/term_cred_att
gen lvl2_prop_comp = term_lvl2_att/term_cred_att_2
gen dev_prop_comp = term_dev_att/term_cred_att_2

foreach v in withdrawn lvl2 dev {
	egen tmp1 = sum(term_`v') if enrolled == 1, by(vccsid)
	if "`v'" == "withdrawn" {
		egen tmp2 = sum(term_cred_att) if enrolled == 1, by(vccsid)
	}
	else {
		egen tmp2 = sum(term_cred_att_2) if enrolled == 1, by(vccsid)
	}
	egen x = max(tmp1), by(vccsid)
	egen y = max(tmp2), by(vccsid)
	gen overall_`v'_prop_comp = x/y
	drop tmp1 tmp2 x y
}
egen tmp1 = sum(term_cred_earn) if enrolled == 1, by(vccsid)
egen tmp2 = sum(term_cred_att_2) if enrolled == 1, by(vccsid)
egen x = max(tmp1), by(vccsid)
egen y = max(tmp2), by(vccsid)
gen overall_prop_comp = x/y
drop tmp1 tmp2 x y
foreach v in prop_comp withdrawn_prop_comp {
	egen `v'_sd = sd(`v'), by(vccsid)
}
drop term_withdrawn term_cred_earn term_cred_att_2 term_lvl2_att term_dev_att
foreach v in term_cred_att term_gpa prop_comp withdrawn_prop_comp lvl2_prop_comp dev_prop_comp {
	replace `v' = 0 if enrolled == 0
}
merge 1:1 vccsid strm using "${main_dir}/intermediate_files/new/temp_11.dta", keep(1 3) nogen
replace repeated = 0 if repeated == .
rename repeated repeat
egen college_entropy_new = max(college_entropy), by(vccsid)
drop college_entropy
rename college_entropy_new college_entropy
preserve
	keep vccsid-grad_1years overall_* *_sd college_entropy
	duplicates drop
	isid vccsid
	sort vccsid
	save "${main_dir}/intermediate_files/new/part_2.dta", replace
restore
drop grad_vccs_6years-grad_1years overall_* *_sd college_entropy
gen pell_0 = (pell == 0)
gen pell_1 = (pell == 1)
drop pell
gen available = 1
* gen mid_term = last_term // (MODIFY HERE!!)
gen flag = int((strm - first_nonde_strm) / 10) + 1
assert flag >= 1 & flag <= 6
tostring flag, replace
gen qt = "fa"
replace qt = "sp" if mod(strm,10) == 2
replace qt = "su" if mod(strm,10) == 3
gen suffix = "_" + qt+flag
drop flag qt first_* last_* strm
reshape wide enrolled-available, i(vccsid) j(suffix) string
sort vccsid
egen available_sum = rowtotal(available_*)
assert available_sum >= 1 & available_sum <= 17
drop available_sum
** Fill in zeros
foreach v in fa sp su {
	forvalues i=1/6 {
		assert available_`v'`i' == 1 if !mi(available_`v'`i')
		replace available_`v'`i' = 0 if mi(available_`v'`i')
	}
}
foreach v in fa sp su {
	forvalues i=1/6 {
		assert enrolled_`v'`i' == . if available_`v'`i' == 0
		replace enrolled_`v'`i' = 0 if enrolled_`v'`i' == .
	}
}
foreach vv in pell_0 pell_1 degree_seeking term_cred_att term_gpa prop_comp withdrawn_prop_comp lvl2_prop_comp dev_prop_comp repeat {
	foreach v in fa sp su {
		forvalues i=1/6 {
			assert `vv'_`v'`i' == . | `vv'_`v'`i' == 0 if enrolled_`v'`i' == 0
			replace `vv'_`v'`i' = 0 if enrolled_`v'`i' == 0
		}
	}
}
save "${main_dir}/intermediate_files/new/part_3.dta", replace


use "${main_dir}/intermediate_files/new/part_2.dta", clear
merge 1:1 vccsid using "${main_dir}/intermediate_files/new/part_1.dta", keep(3) nogen // (MODIFY HERE!!)
merge 1:1 vccsid using "${main_dir}/intermediate_files/new/part_3.dta", keep(1 3) nogen
merge 1:1 vccsid using "${main_dir}/intermediate_files/new/full_data_enrolled_terms.dta", keepusing(valid) keep(1 3) nogen
order vccsid valid first_nonde_strm-grad_1years available_*
sort vccsid
assert first_degree_strm > last_term
foreach v in prop_comp dev_prop_comp lvl2_prop_comp withdrawn_prop_comp {
	rename overall_`v' `v'
}
* Apply log transformation to finaid variables
foreach v in grants sub_loans unsub_loans others {
	foreach t in yr1 yr2 yr3 yr4 yr5 yr6 { // (MODIFY HERE!!)
		gen `v'_`t'_new = log(`v'_`t'+1)
		drop `v'_`t'
		rename `v'_`t'_new `v'_`t'
	}
}
save "${main_dir}/intermediate_files/new/part_123.dta", replace



************************************************
* Finalize term-level and overall NSC predictors
************************************************
use "${main_dir}/intermediate_files/new/temp_nsc_enrl.dta", clear
drop if pre_flag == 1
drop pre_flag
merge m:1 vccsid using "${main_dir}/intermediate_files/new/part_123.dta", keep(3) keepusing(valid) nogen
gen is_summer = (mod(strm,10) == 3)
egen mean_1_tmp = mean(enrl_intensity) if valid == 0 & enrolled_nsc == 1 & is_summer == 0
egen mean_2_tmp = mean(enrl_intensity) if valid == 0 & enrolled_nsc == 1 & is_summer == 1
egen mean_1 = max(mean_1_tmp)
egen mean_2 = max(mean_2_tmp)
drop *_tmp
replace enrl_intensity = mean_1 if enrl_intensity == . & is_summer == 0
replace enrl_intensity = mean_2 if enrl_intensity == . & is_summer == 1
drop valid is_summer mean_1 mean_2
sort vccsid strm college
bys vccsid strm: egen sum_enrl_intensity_nsc = sum(enrl_intensity_nsc)
bys vccsid strm: egen max_enrolled_nsc = max(enrolled_nsc)
preserve
	keep vccsid strm max_enrolled_nsc sum_enrl_intensity_nsc
	rename sum_enrl_intensity_nsc enrl_intensity_nsc
	rename max_enrolled_nsc enrolled_nsc
	sort vccsid strm
	order vccsid strm enrolled_nsc
	duplicates drop
	isid vccsid strm
	save "${main_dir}/intermediate_files/new/temp_14.dta", replace
restore
drop max_enrolled_nsc sum_enrl_intensity_nsc
sort vccsid strm college
drop if enrolled_nsc == 0
collapse (max) enrolled_nsc (sum) enrl_intensity, by(vccsid college)
sort vccsid college
merge m:1 college using "${main_dir}/intermediate_files/new/temp_nsc_quality.dta", keep(1 3) nogen
foreach v in admrate gradrate satvr25 satvr75 satmt25 satmt75 satwr25 satwr75 {
	gen tmp = enrl_intensity_nsc
	replace tmp = 0 if `v' == .
	replace `v' = 0 if `v' == .
	gen y = tmp*`v'
	egen sum1 = sum(tmp), by(vccsid)
	egen sum2 = sum(y), by(vccsid)
	gen new_`v' = sum2/sum1
	drop tmp `v' y sum1 sum2
	rename new_`v' `v'
}
drop enrl_intensity_nsc
sort vccsid college
merge m:1 college using "${main_dir}/intermediate_files/new/temp_nsc_coll_type.dta", keep(1 3) nogen
forvalues i=1/8 {
	egen nsc_coll_type_`i'_new = max(nsc_coll_type_`i'), by(vccsid)
	drop nsc_coll_type_`i'
	rename nsc_coll_type_`i'_new nsc_coll_type_`i'
}
drop college
duplicates drop
isid vccsid
sort vccsid
save "${main_dir}/intermediate_files/new/temp_15.dta", replace


use "${main_dir}/intermediate_files/new/part_123.dta", clear
keep vccsid
merge 1:1 vccsid using "${main_dir}/intermediate_files/new/temp_15.dta", keep(1 3) nogen
replace enrolled_nsc = 0 if enrolled_nsc == .
foreach v in admrate gradrate satvr25 satvr75 satmt25 satmt75 satwr25 satwr75 {
	replace `v' = 0 if enrolled_nsc == 0
}
forvalues i=1/8 {
	replace nsc_coll_type_`i' = 0 if enrolled_nsc == 0
}
sort vccsid
save "${main_dir}/intermediate_files/new/part_4.dta", replace


use "${main_dir}/intermediate_files/new/temp_6.dta", clear
keep vccsid strm
sort vccsid strm
merge m:1 vccsid using "${main_dir}/intermediate_files/new/part_123.dta", keep(2 3) nogen
keep vccsid strm
sort vccsid strm
merge 1:1 vccsid strm using "${main_dir}/intermediate_files/new/temp_14.dta", nogen
sort vccsid strm
foreach v in enrolled_nsc enrl_intensity_nsc {
	replace `v' = 0  if `v' == .
}
sort vccsid strm
bys vccsid: gen flag = int((_n - 1) / 3) + 1 // (MODIFY HERE!!)
assert flag >= 1 & flag <= 6
tostring flag, replace
gen qt = "fa"
replace qt = "sp" if mod(strm,10) == 2
replace qt = "su" if mod(strm,10) == 3
gen suffix = "_" + qt+flag
drop flag qt strm
reshape wide enrolled_nsc enrl_intensity_nsc, i(vccsid) j(suffix) string
sort vccsid
foreach v in fa sp su {
	forvalues i=1/6 {
		assert enrl_intensity_nsc_`v'`i' == . if enrolled_nsc_`v'`i' == .
		assert enrolled_nsc_`v'`i' == . if enrl_intensity_nsc_`v'`i' == .
		replace enrolled_nsc_`v'`i' = 0 if enrolled_nsc_`v'`i' == .
		replace enrl_intensity_nsc_`v'`i' = 0 if enrl_intensity_nsc_`v'`i' == .
	}
}
save "${main_dir}/intermediate_files/new/part_5.dta", replace


****************************************
* Merge everything into the full dataset
****************************************
use "${main_dir}/intermediate_files/new/part_123.dta", clear
merge 1:1 vccsid using "${main_dir}/intermediate_files/new/part_4.dta", nogen
merge 1:1 vccsid using "${main_dir}/intermediate_files/new/part_5.dta", nogen
sort vccsid
drop pell_ind
save "${main_dir}/intermediate_files/new/full_data_truncated.dta", replace


foreach t in fa sp su {
	foreach y in 1 2 3 4 5 6 {	
		gen product_`t'`y' = available_`t'`y' * enrolled_`t'`y'
	}
}

egen num_vccs = rowtotal(product_*)
egen available = rowtotal(available_*)
gen pct_vccs = num_vccs/available

drop num_vccs

foreach v in degree_seeking term_cred_att term_gpa prop_comp withdrawn_prop_comp lvl2_prop_comp dev_prop_comp repeat pell_0 pell_1 enrolled_nsc enrl_intensity_nsc enrolled available product {
	foreach t in fa sp su {
		foreach y in 1 2 3 4 5 6 {
			drop `v'_`t'`y'
		}
	}
}

save "${main_dir}/intermediate_files/new/full_data_truncated_no_term_specific.dta", replace
