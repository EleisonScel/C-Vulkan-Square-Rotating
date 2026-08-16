/* Copyright 2026 EleisonScel
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Portable safe mathematical modulo.
 *
 * Every sa_math_mod_<type> function computes the mathematical modulus of
 * two values of the given type without undefined behaviour, unlike the
 * built-in '%' operator.
 * (negative result is IMPOSSIBLE)
 *
 * Precondition:
 * result_pointer	- isn't NULL
 *
 * Arguments:
 * dividend			- dividend
 * modulus			- size of the residue class ring
 * result_pointer	- pointer to store the remainder
 *
 * Returns:
 * true				- overflow occurred, zero modulus, result pointer is NULL
 * false			- modulus computed successfully
 */

#pragma once

#ifndef SAFE_MODULUS_H
#define SAFE_MODULUS_H

#	include "assert_m.h"/* assert_check_m */

#	include <stdint.h>	/* uint8_t	*/
#	include <stddef.h>	/* size_t	*/
#	include <stdbool.h>	/* bool		*/

#	if defined(UINT8_MAX)
static inline bool sa_math_mod_uint8_t(
		uint8_t dividend, uint8_t modulus, uint8_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus == 0 )
		return true;

	*result_pointer = dividend % modulus;
	return false;
}
#	endif /* UINT8_MAX */

#	if defined(INT8_MIN)
static inline bool sa_math_mod_int8_t(
		int8_t dividend, int8_t modulus, int8_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus == 0 )
		return true;

	if ( modulus == -1 ) {
		*result_pointer = 0;
		return false;
	}

	int8_t result = (int8_t)(dividend % modulus);

	if ( result < 0 )
		result = (modulus < 0) ? (result - modulus) : (result + modulus);

	*result_pointer = result;

	return false;
}
#	endif /* INT8_MIN */

#	if defined(UINT16_MAX)
static inline bool sa_math_mod_uint16_t(
		uint16_t dividend, uint16_t modulus, uint16_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus == 0 )
		return true;

	*result_pointer = dividend % modulus;
	return false;
}
#	endif /* UINT16_MAX */

#	if defined(INT16_MIN)
static inline bool sa_math_mod_int16_t(
		int16_t dividend, int16_t modulus, int16_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus == 0 )
		return true;

	if ( modulus == -1 ) {
		*result_pointer = 0;
		return false;
	}

	int16_t result = (int16_t)(dividend % modulus);

	if ( result < 0 )
		result = (modulus < 0) ? (result - modulus) : (result + modulus);

	*result_pointer = result;

	return false;
}
#	endif /* INT16_MIN */

#	if defined(UINT32_MAX)
static inline bool sa_math_mod_uint32_t(
		uint32_t dividend, uint32_t modulus, uint32_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus == 0 )
		return true;

	*result_pointer = dividend % modulus;
	return false;
}
#	endif /* UINT32_MAX */

#	if defined(INT32_MIN)
static inline bool sa_math_mod_int32_t(
		int32_t dividend, int32_t modulus, int32_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus == 0 )
		return true;

	if ( modulus == -1 ) {
		*result_pointer = 0;
		return false;
	}

	int32_t result = (int32_t)(dividend % modulus);

	if ( result < 0 )
		result = (modulus < 0) ? (result - modulus) : (result + modulus);

	*result_pointer = result;

	return false;
}
#	endif /* INT32_MIN */

#	if defined(UINT64_MAX)
static inline bool sa_math_mod_uint64_t(
		uint64_t dividend, uint64_t modulus, uint64_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus == 0 )
		return true;

	*result_pointer = dividend % modulus;
	return false;
}
#	endif /* UINT64_MAX */

#	if defined(INT64_MIN)
static inline bool sa_math_mod_int64_t(
		int64_t dividend, int64_t modulus, int64_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus == 0 )
		return true;

	if ( modulus == -1 ) {
		*result_pointer = 0;
		return false;
	}

	int64_t result = (int64_t)(dividend % modulus);

	if ( result < 0 )
		result = (modulus < 0) ? (result - modulus) : (result + modulus);

	*result_pointer = result;

	return false;
}
#	endif /* INT64_MIN */

static inline bool sa_math_mod_size_t(
		size_t dividend, size_t modulus, size_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus == 0 )
		return true;

	*result_pointer = dividend % modulus;
	return false;
}

static inline bool sa_math_mod_ptrdiff_t(
		ptrdiff_t dividend, ptrdiff_t modulus, ptrdiff_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus == 0 )
		return true;

	if ( modulus == -1 ) {
		*result_pointer = 0;
		return false;
	}

	ptrdiff_t result = (ptrdiff_t)(dividend % modulus);

	if ( result < 0 )
		result = (modulus < 0) ? (result - modulus) : (result + modulus);

	*result_pointer = result;

	return false;
}

static inline bool sa_math_mod_uintmax_t(
		uintmax_t dividend, uintmax_t modulus, uintmax_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus == 0 )
		return true;

	*result_pointer = dividend % modulus;
	return false;
}

static inline bool sa_math_mod_intmax_t(
		intmax_t dividend, intmax_t modulus, intmax_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus == 0 )
		return true;

	if ( modulus == -1 ) {
		*result_pointer = 0;
		return false;
	}

	intmax_t result = (intmax_t)(dividend % modulus);

	if ( result < 0 )
		result = (modulus < 0) ? (result - modulus) : (result + modulus);

	*result_pointer = result;

	return false;
}

#	ifdef UINTPTR_MAX
static inline bool sa_math_mod_uintptr_t(
		uintptr_t dividend, uintptr_t modulus, uintptr_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus == 0 )
		return true;

	*result_pointer = dividend % modulus;
	return false;
}
#	endif /* UINTPTR_MAX */

#	ifdef INTPTR_MIN
static inline bool sa_math_mod_intptr_t(
		intptr_t dividend, intptr_t modulus, intptr_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus == 0 )
		return true;

	if ( modulus == -1 ) {
		*result_pointer = 0;
		return false;
	}

	intptr_t result = (intptr_t)(dividend % modulus);

	if ( result < 0 )
		result = (modulus < 0) ? (result - modulus) : (result + modulus);

	*result_pointer = result;

	return false;
}
#	endif /* INTPTR_MIN */

#endif /* SAFE_MODULUS_H */
