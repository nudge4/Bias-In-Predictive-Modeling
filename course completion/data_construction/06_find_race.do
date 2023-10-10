*Name: create_merged_student_demographics.do
*Author: Yifeng Song
*Last Modified: 08/23/2020
*Purpose: Merge the VCCS Student files, extract demographic information of each student, 
*which will be used as predictors in building grade prediction models. (Update: We exclude
* race/ethnicity/gender from grade prediction models, and only keep age)


** Directory globals
global data "D:/Yifeng -- Project Work/ys8mz_sandbox"
global sample_vccsid_list "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\all_vccsid.dta"


**************************************************************************************
** Merge and process the Student files to retrieve demographic information of students
**************************************************************************************
local files: dir "${data}/Build/Student" files "*.dta.zip"
local i = 0
foreach file in `files' {
	if "`file'" != "student_all_records.dta.zip" {
		if `i' == 0 {
			zipuse "${data}/Build/Student/`file'", clear
			keep vccsid strm age gender new_race
			gen source = `i'
			merge m:1 vccsid using "${sample_vccsid_list}", keep(3) nogen // only keep the observations for students included in the study sample
		}
		else {
			preserve
				zipuse "${data}/Build/Student/`file'", clear
				keep vccsid strm age gender new_race
				gen source = `i'
				merge m:1 vccsid using "${sample_vccsid_list}", keep(3) nogen // only keep the observations for students included in the study sample
				tempfile stu_temp_`i'
				save "`stu_temp_`i''", replace
			restore
			append using "`stu_temp_`i''", force
		}
		di "`file'"
		local i = `++i'
	}
}
drop if strm > 2202
replace strm = 2003 if strm == .
gsort vccsid -source // relying on Student files during the more recent terms to determine the demographic information if there're discrepancies
egen max_source = max(source), by(vccsid)
keep if source == max_source
duplicates drop
drop source max_source

egen gender_new = mode(gender), by(vccsid)
gen male = (gender_new == "M")
drop gender gender_new
gen new_race_copy = new_race
replace new_race_copy = . if inlist(new_race_copy, 0, 7)
egen new_race_new = mode(new_race_copy), by(vccsid)
gen tmp = 0
replace tmp = 1 if new_race_copy == .
egen flag = min(tmp), by(vccsid)
replace new_race_new = 8 if flag == 0 & new_race_new == .
gen white = (new_race_new == 1)
gen afam = (new_race_new == 2)
gen hisp = (new_race_new == 3)
gen asian = (new_race_new == 4)
gen other = inlist(new_race_new,5,6,8)
drop new_race new_race_copy new_race_new tmp flag
duplicates drop
sort vccsid age
bys vccsid: keep if _n == 1
isid vccsid
sort vccsid
rename strm age_strm
drop age age_strm
gen race = "other"
replace race = "white" if white == 1
replace race = "afam" if afam == 1
replace race = "hisp" if hisp == 1
replace race = "asian" if asian == 1
save "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\race.dta", replace
