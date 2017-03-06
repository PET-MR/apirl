/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-config.h.in                                                 *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : (X)MedCon template configuration header (configure)      *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-config.h.in,v 1.22 2010/08/28 23:44:23 enlf Exp $
 */

/*
   Copyright (C) 1997-2010 by Erik Nolf

   This program is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the
   Free Software Foundation; either version 2, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
   Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   59 Place - Suite 330, Boston, MA 02111-1307, USA.  */

#pragma once

/* Define if format enabled */
#define MDC_INCLUDE_INTF  1

/* Define decompression program */
#define MDC_DECOMPRESS  "@DECOMPRESS@"

/* Define GLIB related stuff */
#define GLIBSUPPORTED 0

/* Define some version variables */
#define XMEDCON_MAJOR    "@XMEDCON_MAJOR@"
#define XMEDCON_MINOR    "@XMEDCON_MINOR@"
#define XMEDCON_MICRO    "@XMEDCON_MICRO@"
#define XMEDCON_PRGR     "@XMEDCON_PRGR@"
#define XMEDCON_DATE     "@XMEDCON_DATE@"
#define XMEDCON_VERSION  "@XMEDCON_VERSION@"
#define XMEDCON_LIBVERS  "@XMEDCON_LIBVERS@"

/* Define if format enabled */
// pOR AHORA SOLO USO INTEFILE
#define MDC_INCLUDE_ACR   0
#define MDC_INCLUDE_GIF   0
#define MDC_INCLUDE_INW   0
#define MDC_INCLUDE_ANLZ  0
#define MDC_INCLUDE_CONC  0
#define MDC_INCLUDE_ECAT  0
#define MDC_INCLUDE_INTF  1
#define MDC_INCLUDE_DICM  0
#define MDC_INCLUDE_PNG   0
#define MDC_INCLUDE_NIFTI 0
#define MDC_INCLUDE_TPC   0   /* TPC ecat7 write */

/* Por defecto defino los de amd64 */
#define MDC_SIZEOF_SHORT     2
#define MDC_SIZEOF_INT       4
#define MDC_SIZEOF_LONG      8
#define MDC_SIZEOF_LONG_LONG 8
	
/* Define some machine dependencies */
#ifdef x86_64
//	#define MDC_WORDS_BIGENDIAN  @mdc_cv_bigendian@
	#define MDC_SIZEOF_SHORT     2
	#define MDC_SIZEOF_INT       4
	#define MDC_SIZEOF_LONG      8
	#define MDC_SIZEOF_LONG_LONG 8
#else
	#ifdef X86
//		#define MDC_WORDS_BIGENDIAN  @mdc_cv_bigendian@
		#define MDC_SIZEOF_SHORT     2
		#define MDC_SIZEOF_INT       4
		#ifndef MDC_SIZEOF_LONG
		#define MDC_SIZEOF_LONG      4
		#endif
		#define MDC_SIZEOF_LONG_LONG 8
	#endif
#endif



#if MDC_ENABLE_LONG_LONG
#define MDC_SIZEOF_LONG_LONG @ac_cv_sizeof_long_long@
#endif
