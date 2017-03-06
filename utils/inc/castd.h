#pragma once

/**
casSTD:

casper's "standard" utilities. likely c++>=11.
@author Casper da Costa-Luis <casper.dcl@physics.org>
@year 2016
*/

#include <string>
#include <array>
#include <algorithm>
#include <functional>
#include <type_traits>

namespace cas
{

template
	< typename RetType
	, size_t N
	, typename = typename std::enable_if
		<std::is_floating_point<RetType>::value
		|| std::is_integral<RetType>::value
		>::type
	>
/**
Convert "{x,y,...}" to an integer/float array
@author Casper da Costa-Luis <casper.dcl@physics.org>
*/
std::array<RetType, N> str2coords(std::string aux)
{
	std::array<RetType, N> res;

	switch (aux.find("{", 0))
	{
	case string::npos:
		break;
	case 0:
		aux = aux.substr(1, aux.length() - 2);
	}

	size_t pos0 = 0;
	size_t pos1 = aux.find(",", pos0);
	for (int i = 0; i < N - 1; ++i)
	{
		string num = aux.substr(pos0, pos1 - pos0);
		if (std::is_integral<RetType>::value)
			res[i] = static_cast<RetType>(atoll(num.c_str()));
		else
			res[i] = static_cast<RetType>(atof(num.c_str()));
		pos0 = ++pos1;
		pos1 = aux.find(",", pos1);
	}

	switch (aux.find("}", pos0))
	{
	case string::npos:
		pos1 = aux.length();
	}

	string num = aux.substr(pos0, pos1 - pos0);
	if (std::is_integral<RetType>::value)
		res[N] = static_cast<RetType>(atoll(num.c_str()));
	else
		res[N] = static_cast<RetType>(atof(num.c_str()));

	return res;
}

}