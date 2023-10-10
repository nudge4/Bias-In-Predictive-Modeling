use "D:\\Yifeng -- Project Work\\ys8mz_sandbox\\Merged_Class.dta", clear
* keep if college == "Piedmont Virginia"
drop if strm >= 2202



*************************
** Clean up course grades
*************************
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
drop if credit == 0


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


keep acad_group-vccsid est_grade sum_effective_credit sum_grade_point
duplicates drop
sort vccsid strm subject course_num
duplicates tag vccsid strm subject course_num, gen(dups)
rename sum_effective_credit effective_credit
rename sum_grade_point grade_point
** Merge the remaining stuid x term x subject x course_num pairs that still have multiple observations
bys vccsid strm subject course_num: egen sum_effective_credit = sum(effective_credit)
bys vccsid strm subject course_num: egen sum_grade_point = sum(grade_point)
gen new_est_grade = sum_grade_point/sum_effective_credit if dups >= 1
replace est_grade = new_est_grade if dups >= 1
keep vccsid strm college subject course_num est_grade sum_grade_point sum_effective_credit
duplicates drop
isid vccsid strm college subject course_num
drop if sum_effective_credit == 0
rename sum_effective_credit effective_credit
rename sum_grade_point grade_point
foreach t in 2142 2143 2144 2152 2153 2154 2162 2163 2164 2172 2173 2174 2182 2183 2184 2192 2193 2194 {
	preserve
		keep if strm < `t' & strm >= `t'-50
		bys college subject course_num: egen sum_effective_credit = sum(effective_credit)
		bys college subject course_num: egen sum_grade_point = sum(grade_point)
		gen avg_grade = sum_grade_point/sum_effective_credit
		egen sum_effective_credit_2 = sum(effective_credit), by(college)
		egen sum_grade_point_2 = sum(grade_point), by(college)
		gen avg_grade_2 = sum_grade_point_2/sum_effective_credit_2
		keep college subject course_num avg_grade avg_grade_2
		duplicates drop
		isid college subject course_num
		gen strm = `t'
		tempfile avg_`t'
		save "`avg_`t''", replace
	restore
}

clear
foreach t in 2142 2143 2144 2152 2153 2154 2162 2163 2164 2172 2173 2174 2182 2183 2184 2192 2193 2194 {
	append using "`avg_`t''", force
}
gen course = subject + "_" + course_num
drop subject course_num
isid college course strm
sort college course strm
save "C:\\Users\\ys8mz\\Box Sync\\Clickstream\\data\\bias\\avg_grades.dta", replace
