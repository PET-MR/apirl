/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-color.h                                                     *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : m-color.c header file                                    *
 *                                                                         * 
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-color.h,v 1.14 2010/08/28 23:44:23 enlf Exp $
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

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/
int  MdcLoadLUT(const char *lutname);
void MdcGrayScale(Uint8 *palette);
void MdcInvertedScale(Uint8 *palette);
void MdcRainbowScale(Uint8 *palette);
void MdcCombinedScale(Uint8 *palette);
void MdcHotmetalScale(Uint8 *palette);
void MdcGetColorMap(int map, Uint8 palette[]);
int MdcSetPresentMap(Uint8 palette[]);
