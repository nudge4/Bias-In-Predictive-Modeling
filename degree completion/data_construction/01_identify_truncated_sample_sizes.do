/* This script finds out the distribution of currently enrolled students at VCCS in terms of enrollment lengths (how many terms have elapsed since the first term at VCCS). */

*** Creating sample for analysis and outcomes of interest ***

global user = c(username) 

global box = "/Users/$user/Box Sync"

global gitrepo = "$box/GitHub/predictive_modeling_degree_completion_aera_open"

global working_data = "$box/VCCS restricted student data/Master_student_x_term_data"
	
global project_data = "$box/Predictive Models of College Completion (VCCS)/dta/new"
	
use "$working_data/all_term_level_restricted.dta", clear


*** Keeping only students who were enrolled at least one term in 
*** a college-level curriculum
keep if any_abv_collev_nonde==1

keep if first_nonde_strm >= 2134 & first_nonde_strm <=2194


*************************************
*** Creating outcomes of interest ***
*************************************

*** Window of observation: 6 years (18 terms) from first_nonde_strm
gen window_obs = term_consec >= first_nonde_term_consec & ///
	term_consec < (first_nonde_term_consec + 18)
	
*** Earned degree within 6 years
*** 	(max) in collapse
gen grad_vccs_6years = grad_vccs if window_obs==1
gen grad_nonvccs_6years = grad_nonvccs if window_obs==1

*** Term earned first degree
***		(min) in collapse
gen first_degree_strm = strm if grad_vccs_6years==1 | grad_nonvccs_6years==1

*** Enrolled at non-VCCS within 6 years
*** 	(max) in collapse
gen enr_nonvccs_6years = enr_nonvccs if window_obs==1

*** First term enrolled at non-VCCS
***		(min) in collapse
gen first_enr_nonvccs_strm = strm if enr_nonvccs_6years==1

*** Last strm within window of observation
gen last_strm = ""
	// Spring 20XX to Fall (20XX+5)
	replace last_strm = "2"+string(real(substr(string(first_nonde_strm),2,2))+5)+"4" ///
		if substr(string(first_nonde_strm),4,1)=="2"
	// Summer 20XX to Spring (20XX+6)
	replace last_strm = "2"+string(real(substr(string(first_nonde_strm),2,2))+6)+"2" ///
		if substr(string(first_nonde_strm),4,1)=="3"	
	// Fall 20XX to Summer (20XX+6)
	replace last_strm = "2"+string(real(substr(string(first_nonde_strm),2,2))+6)+"3" ///
		if substr(string(first_nonde_strm),4,1)=="4"	
	destring last_strm, replace	
	
*** Collapsing to student-level
collapse (max) grad_vccs_6years grad_nonvccs_6years enr_nonvccs_6years ///
		(min) first_degree_strm first_enr_nonvccs_strm ///
	, by(vccsid first_nonde_* last_strm)

*** Merging in student-level data on all degree levels earned
	preserve
		zipuse "$working_data/student_level_vccs_degree_type_strm.dta.zip", clear
		tempfile tmp
		save "`tmp'", replace
	restore
	merge 1:1 vccsid using "`tmp'"
	* zipmerge 1:1 vccsid using "$working_data/student_level_vccs_degree_type_strm.dta.zip"
		drop if _merge==2
		foreach type in associate certificate diploma {
			* Filling in zeros for students not present in VCCS graduation data
			replace deg_vccs_`type' = 0 if _merge==1
			
			* Not including degrees earned outside 6 years 
			replace deg_vccs_`type' = 0 if ///
				deg_vccs_`type'_strm > last_strm & ///
				deg_vccs_`type'_strm!=. 
			replace deg_vccs_`type'_strm = . if ///
			deg_vccs_`type'_strm > last_strm
			
			* Checking degree timing 
			assert deg_vccs_`type'_strm >= first_nonde_strm & ///
				deg_vccs_`type'_strm <= last_strm if ///
				deg_vccs_`type'==1
			}
		drop _merge
		
	preserve
		zipuse "$working_data/student_level_nonvccs_degree_type_strm.dta.zip", clear
		tempfile tmp
		save "`tmp'", replace
	restore
	merge 1:1 vccsid using "`tmp'"
	*zipmerge 1:1 vccsid using "$working_data/student_level_nonvccs_degree_type_strm.dta.zip"
		drop if _merge==2
		foreach type in graduate bachelor associate certificate diploma unknown {
			* Filling in zeros for students not present in nonVCCS graduation data
			replace deg_nonvccs_`type' = 0 if _merge==1
			
			* Not including degrees earned outside 6 years 
			replace deg_nonvccs_`type' = 0 if ///
				deg_nonvccs_`type'_strm > last_strm & ///
				deg_nonvccs_`type'_strm!=. 
			replace deg_nonvccs_`type'_strm = . if ///
				deg_nonvccs_`type'_strm > last_strm & ///
				deg_nonvccs_`type'_strm!=. 
				
			* Checking degree timing
			drop if (deg_nonvccs_`type'_strm >= first_nonde_strm & deg_nonvccs_`type'_strm <= last_strm) == 0 & deg_nonvccs_`type'==1
			/*
			assert deg_nonvccs_`type'_strm >= first_nonde_strm & ///
				deg_nonvccs_`type'_strm <= last_strm if ///
				deg_nonvccs_`type'==1
			*/
			}
		drop _merge

*** Saving data
	isid vccsid 
	save "$project_data/sample_and_outcomes_related_to_current_academic_year.dta", replace


use "$project_data/sample_and_outcomes_related_to_current_academic_year", clear
merge 1:m vccsid using "D:\Yifeng -- Project Work\ys8mz_sandbox\Merged_Class.dta", keep(1 3) nogen
keep vccsid strm
duplicates drop
preserve
	use "D:\Yifeng -- Project Work\ys8mz_sandbox\Merged_GPA.dta", clear
	drop if unt_taken_prgrss == 0
	drop if strm < 2034 | strm > 2194
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

foreach t in 2192 2193 2194 {
	preserve
		keep if strm == `t'
		sort vccsid strm
		bys vccsid: keep if _n == 1
		keep vccsid strm
		duplicates drop
		sort vccsid
		merge 1:1 vccsid using "$project_data/sample_and_outcomes_related_to_current_academic_year.dta", keep(3) nogen
		qui count if first_degree_strm < `t'
		di r(N)/_N
		qui count if first_degree_strm == `t'
		di r(N)/_N
		drop if first_degree_strm <= `t'
		drop if first_nonde_strm > `t'
		keep vccsid strm first_nonde_strm first_degree_strm
		sort vccsid
		rename strm crnt_strm 
		gen yr = int((crnt_strm-first_nonde_strm)/10)
		gen term_diff = crnt_strm - first_nonde_strm - 10*yr
		replace term_diff = 1 if term_diff == 8
		replace term_diff = 2 if term_diff == 9
		gen nth_term = 3*yr + term_diff + 1
		keep if nth_term <= 17
		keep vccsid crnt_strm nth_term
		isid vccsid
		sort vccsid
		tempfile nth_`t'
		save "`nth_`t''", replace
	restore
}
clear
foreach t in 2192 2193 2194 {
	append using "`nth_`t''", force
}
tab nth_term if crnt_strm == 2192
tab nth_term if crnt_strm == 2193
tab nth_term if crnt_strm == 2194
save "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\enrollment_lengths_by_crnt_term.dta", replace

bys nth_term: gen counts = _N
keep nth_term counts
duplicates drop
sort nth_term
egen total_counts = sum(counts)
gen prop = counts/total_counts
drop *counts
order nth_term prop

preserve
	use "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\enrolled_nth.dta", clear
	keep vccsid first_nonde_strm
	duplicates drop
	qui count if first_nonde_strm >= 2123
	local N2 = r(N)
	qui count if first_nonde_strm < 2123
	local N1 = r(N)
restore
gen train_sample_size = round(prop * `N1') // This number could be found using the files "full_data_enrolled_terms.dta" and "enrolled_nth.dta"
egen total_sample_size = sum(train_sample_size)
replace train_sample_size = train_sample_size + (`N1' - total_sample_size) if _n == 1
drop total*
gen valid_sample_size = round(prop * `N2') // This number could be found using the files "full_data_enrolled_terms.dta" and "enrolled_nth.dta"
egen total_sample_size = sum(valid_sample_size)
replace valid_sample_size = valid_sample_size + (`N2' - total_sample_size) if _n == 1
drop total*

sort nth_term
save "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\truncation_sample_sizes.dta", replace
