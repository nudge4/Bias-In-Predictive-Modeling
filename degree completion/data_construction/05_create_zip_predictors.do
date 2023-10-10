/* This script creates a special type of demographic predictors: The predictors that are related to zip codes. They include student's distance to college, median households income associated with the zcta x year, percentage below poverty associated with the zcta x year. The missing value indicators showing the different types of the missing values are also generated. */

clear
global data "D:/Yifeng -- Project Work/ys8mz_sandbox"

***************************************************
** Create the parents' highest education predictors
***************************************************
local files: dir "${data}/Build/Student" files "*.dta.zip"
local i = 0
foreach file in `files' {
	preserve
		zipuse "${data}/Build/Student/`file'", clear
		merge m:1 vccsid using "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\vccsid_list.dta", keep(3) nogen
		keep vccsid strm total_credit_hrs stud_zcta
		gsort vccsid strm -total_credit_hrs
		bys vccsid strm: keep if _n == 1
		tempfile stu_temp_`i'
		save "`stu_temp_`i''", replace
	restore
	append using "`stu_temp_`i''", force
	di "`file'"
	local i = `++i'
}

keep vccsid strm stud_zcta
sort vccsid strm

gen zcta_mi_type = mi(stud_zcta)
egen tmp = mode(stud_zcta), by(vccsid)
replace stud_zcta = tmp if mi(stud_zcta) | length(stud_zcta) < 5
rename stud_zcta zcta
sort vccsid strm
drop tmp
bys vccsid: gen tmp = zcta[_n-1] if _n > 1
replace zcta = tmp if mi(zcta)
drop tmp
bys vccsid: gen indx = _n
replace indx = . if mi(zcta)
egen min_indx = min(indx), by(vccsid)
gen tmp = zcta if min_indx == indx
egen tmp2 = mode(tmp), by(vccsid)
replace zcta = tmp2 if mi(zcta)
drop indx min_indx tmp tmp2
replace zcta_mi_type = 2 if mi(zcta)

save "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\all_zcta.dta", replace


use "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\full_data_truncated.dta", clear
rename last_term strm
merge 1:1 vccsid strm using "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\all_zcta.dta"
keep vccsid strm zcta zcta_mi_type _merge
sort vccsid strm
bys vccsid: replace zcta = zcta[_n-1] if mi(zcta)
gsort vccsid -strm
bys vccsid: replace zcta = zcta[_n-1] if mi(zcta)
sort vccsid strm
drop if _merge == 2

replace zcta = "00000" if mi(zcta)
tostring strm, gen(strm_str)
gen year = substr(strm_str, 1, 1) + "0" + substr(strm_str, 2, 2)
destring year, replace
drop strm_str
merge m:1 zcta year using "C:/Users/ys8mz/Downloads/zcta_poverty_median_income_enlarged_2.dta", keep(1 3) nogen

replace perc_below_pov_orig = . if zcta_mi_type == 1 & perc_below_pov_mi_type == 0
replace median_income_orig = . if zcta_mi_type == 1 & median_income_mi_type == 0

replace perc_below_pov_mi_type = zcta_mi_type * 2 + perc_below_pov_mi_type
replace perc_below_pov_mi_type = perc_below_pov_mi_type + 1 if zcta_mi_type == 1
replace perc_below_pov_mi_type = 6 if zcta_mi_type == 2
replace median_income_mi_type = zcta_mi_type * 2 + median_income_mi_type
replace median_income_mi_type = median_income_mi_type + 1 if zcta_mi_type == 1
replace median_income_mi_type = 6 if zcta_mi_type == 2

keep vccsid strm median_income_households perc_below_pov median_income_mi_type perc_below_pov_mi_type median_income_orig perc_below_pov_orig
order vccsid strm median_income_households perc_below_pov median_income_mi_type perc_below_pov_mi_type median_income_orig perc_below_pov_orig
sort vccsid strm
isid vccsid strm
egen tmp1 = mean(median_income_households)
replace median_income_households = tmp1 if median_income_households == .
egen tmp2 = mean(perc_below_pov)
replace perc_below_pov = tmp2 if perc_below_pov == .
drop tmp1 tmp2

rename strm last_term
save "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\zip_part1.dta", replace


clear
global data "D:/Yifeng -- Project Work/ys8mz_sandbox"

***************************************************
** Create the parents' highest education predictors
***************************************************
local files: dir "${data}/Build/Student" files "*.dta.zip"
local i = 0
foreach file in `files' {
	preserve
		zipuse "${data}/Build/Student/`file'", clear
		keep vccsid strm total_credit_hrs distance_stud_to_coll
		gsort vccsid strm -total_credit_hrs
		bys vccsid strm: keep if _n == 1
		tempfile stu_temp_`i'
		save "`stu_temp_`i''", replace
	restore
	append using "`stu_temp_`i''", force
	di "`file'"
	local i = `++i'
}

keep vccsid strm distance_stud_to_coll
rename strm last_term
sort vccsid last_term
rename distance_stud_to_coll distance
gen distance_orig = distance
gen distance_mi_type = mi(distance)
bys vccsid: replace distance = distance[_n-1] if mi(distance)
gsort vccsid -last_term
bys vccsid: replace distance = distance[_n-1] if mi(distance)
replace distance_mi_type = 2 if mi(distance)

merge 1:1 vccsid last_term using "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\full_data_truncated.dta", keep(2 3) nogen
keep vccsid last_term distance distance_mi_type distance_orig

rename distance distance_stud_to_coll
egen mean_dist = mean(distance_stud_to_coll)
replace distance_stud_to_coll = mean_dist if distance_stud_to_coll == .
drop mean_dist
merge 1:1 vccsid last_term using "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\zip_part1.dta", keep(1 3) nogen
rename distance_stud_to_coll distance
sort vccsid last_term
save "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\zip_new.dta", replace
