/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-raw.h                                                       *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : m-raw.c header file                                      *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-raw.h,v 1.14 2010/08/28 23:44:23 enlf Exp $
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


#ifndef __M_RAW_H__
#define __M_RAW_H__

/****************************************************************************
                              D E F I N E S 
****************************************************************************/

#define MdcReadInterActive(a)		MdcReadRAW(a)

typedef struct MdcRawInputStruct_t {

  Uint32 gen_offset, img_offset;
  Uint32 *abs_offset;
  Int8 DIFF, REPEAT;

}MdcRawInputStruct;

typedef struct MdcRawPrevInputStruct_t {

  Uint32 XDIM, YDIM, NRIMGS;
  Uint32 GENHDR, IMGHDR, ABSHDR;
  Int16  PTYPE;
  Int8   DIFF, HDRREP, PSWAP;

}MdcRawPrevInputStruct;

extern MdcRawInputStruct mdcrawinput;
extern MdcRawPrevInputStruct mdcrawprevinput;

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/
void MdcInitRawPrevInput(void);
char *MdcReadRAW(FILEINFO *fi);
char *MdcWriteRAW(FILEINFO *fi);
int MdcCheckPredef(const char *fname);
char *MdcReadPredef(const char *fname);
char *MdcWritePredef(const char *fname);

#endif
