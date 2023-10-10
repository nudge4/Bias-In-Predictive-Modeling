clear
forvalues y=2008/2019 {
	foreach v in "3_spe" "4_sue" "6_fae" {
		preserve
			zipuse "C:\\Users\\ys8mz\\Box Sync\\VCCS restricted student data\\Build\\Course\\Course_`y'_`v'.dta.zip", clear
			keep vccsid_instructor college strm
			duplicates drop
			merge m:1 vccsid_instructor college using "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\instructor_list.dta", keep(3) nogen
			tempfile tmp
			save "`tmp'", replace
		restore
		append using "`tmp'", force
	}
}
egen first_strm = min(strm), by(vccsid_instructor college)
keep vccsid_instructor college first_strm
duplicates drop
merge 1:m vccsid_instructor college using "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\instructors.dta", nogen
drop course
duplicates drop
* replace first_strm = 2212 if first_strm == .
assert first_strm <= teaching_strm
gen tenure = teaching_strm - first_strm >= 60
drop first_strm section
duplicates drop
merge 1:m vccsid_instructor college teaching_strm using "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\instructors.dta", nogen
sort vccsid_instructor college teaching_strm course
save "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\instructors.dta", replace


clear
forvalues y=2008/2019 {
	foreach v in "3_spe" "4_sue" "6_fae" {
		preserve
			zipuse "C:\\Users\\ys8mz\\Box Sync\\VCCS restricted student data\\Build\\Course\\Course_`y'_`v'.dta.zip", clear
			keep vccsid_instructor college strm subject course_num section
			duplicates drop
			* keep if college == "Piedmont Virginia"
			merge m:1 vccsid_instructor college using "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\instructor_list.dta", keep(3) nogen
			tempfile tmp
			save "`tmp'", replace
		restore
		append using "`tmp'", force
	}
}
* drop if strm == 2202
gen course = subject + "_" + course_num

foreach t in 2142 2143 2144 2152 2153 2154 2162 2163 2164 2172 2173 2174 2182 2183 2184 2192 2193 2194 {
	preserve
		use "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\instructors.dta", clear
		drop tenure section
		duplicates drop
		keep if teaching_strm == `t'
		isid vccsid_instructor college course
		tempfile tmp
		save "`tmp'", replace
	restore
	preserve
		merge m:1 vccsid_instructor college course using "`tmp'", keep(3) nogen
		drop if strm >= teaching_strm
		tempfile tmp_`t'
		save "`tmp_`t''", replace
	restore
}
clear
foreach t in 2142 2143 2144 2152 2153 2154 2162 2163 2164 2172 2173 2174 2182 2183 2184 2192 2193 2194 {
	append using "`tmp_`t''", force
}

isid vccsid_instructor college teaching_strm strm course section
order vccsid_instructor college teaching_strm strm course section
save "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\taught_before.dta", replace



use "D:\\Yifeng -- Project Work\\ys8mz_sandbox\\Merged_Class.dta", clear
drop if strm >= 2202

preserve
	use "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\taught_before.dta", clear
	keep strm college subject course_num section
	duplicates drop
	* gen college = "Piedmont Virginia"
	tempfile taught
	save "`taught'", replace
restore
merge m:1 strm college subject course_num section using "`taught'", keep(2 3) nogen


replace course_num = substr(course_num, 2, .) if regexm(course_num,"^[0-9][0-9][0-9][0-9]L$")
replace course_num = substr(course_num, 1, 3) if regexm(course_num,"^[0-9][0-9][0-9]L$")
** There are courses with multiple sessions within the same student x college x term: if one session has a non-missing grade but the rest of the sessions have missing grades,
** then we can fill in those missing grades with the grade of the non-missing session of the course
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
** There are courses with multiple sessions within the same student x term (could be across multiple colleges): if one session has a non-missing grade but the rest of the sessions have missing grades,
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



************************************************************************************************************************************************************************
** Next aggregate multiple observations of courses (credits, course grades) so that there's a single observation of course grade per stuid x term x subject x course_num
************************************************************************************************************************************************************************
gen simplified_grade = grade
replace simplified_grade = "UNKNOWN" if inlist(grade, "", "*", "K", "N", "Z", "XN")
replace simplified_grade = "UNKNOWN" if inlist(grade, "R", "I", "X", "XY")
gen effective_credit = credit // effective_credit is used for estimating the new course grades if there're multiple observations per stuid x term x subject x course_num
replace effective_credit = 0 if simplified_grade == "UNKNOWN" | simplified_grade == "W"
gen numeric_grade = 0
replace numeric_grade = 4 if inlist(grade, "A", "S", "P") // assume the courses with grades "S"/"P" is worth 4 grade points just as "A" courses
replace numeric_grade = 3 if grade == "B"
replace numeric_grade = 2 if grade == "C"
replace numeric_grade = 1 if grade == "D"
gen grade_point = numeric_grade*effective_credit
sort vccsid strm subject course_num
bys vccsid strm subject course_num: egen sum_effective_credit = sum(effective_credit)
bys vccsid strm subject course_num: egen sum_grade_point = sum(grade_point)
gen est_grade = sum_grade_point/sum_effective_credit // If there are multiple observations per stuid x term x subject x course_num, we need to calculate a single course grade for it


************************************
** Further Cleanup and Aggregration
************************************
** Clean up the case where for the same stuid x term x subject x course_num, there are multiple observations from different colleges (choose the college in which the course has higher credit and grade).
** So that there're a unique college corresponding to each stuid x term x subject x course_num
set seed 1234
gen r_num = runiform() // Use random number generator to break ties
gsort vccsid strm subject course_num -credit -numeric_grade r_num


*** Keep useful variables, drop duplicates so that the table is unique per student x term x course
keep vccsid strm college subject course_num est_grade sum_effective_credit* sum_grade_point*
duplicates drop
drop if mi(vccsid)
isid vccsid strm college subject course_num
sort vccsid strm college subject course_num
foreach v in effective_credit grade_point {
	rename sum_`v' `v'
}
sort vccsid strm subject course_num
duplicates tag vccsid strm subject course_num, gen(dups)
** Merge the remaining stuid x term x subject x course_num pairs that still have multiple observations
bys vccsid strm subject course_num: egen sum_effective_credit = sum(effective_credit)
bys vccsid strm subject course_num: egen sum_grade_point = sum(grade_point)
gen new_est_grade = sum_grade_point/sum_effective_credit if dups >= 1
replace est_grade = new_est_grade if dups >= 1
drop dups new_est_grade effective_credit* grade_point*
duplicates drop
isid vccsid strm college subject course_num
sort vccsid strm college subject course_num
drop if sum_effective_credit == 0
drop sum_*
sort strm subject course_num vccsid
bys strm college subject course_num: egen avg_grade = mean(est_grade)
keep strm college subject course_num avg_grade
duplicates drop


merge 1:m strm college subject course_num using "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\taught_before.dta", keep(2 3) nogen
bys vccsid_instructor college teaching_strm course: egen past_avg_grade = mean(avg_grade)
keep vccsid_instructor college teaching_strm course past_avg_grade
duplicates drop
merge 1:m vccsid_instructor college teaching_strm course using "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\instructors.dta", keep(2 3) nogen
gen has_past_avg_grade = !mi(past_avg_grade)
replace past_avg_grade = 0 if mi(past_avg_grade)
rename teaching_strm strm
merge 1:m strm college course section using "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\all_observations.dta", keep(2 3) nogen
keep vccsid strm college course section tenure has_past_avg_grade past_avg_grade vccsid_instructor
merge m:1 strm college course section using "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\full_time.dta", nogen
replace has_past_avg_grade = 0 if mi(has_past_avg_grade)
replace past_avg_grade = 0 if mi(past_avg_grade)
replace tenure = 0 if mi(tenure)
drop vccsid_instructor
sort strm college course section vccsid
drop if mi(vccsid)
save "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\instructor_related_predictors.dta", replace
