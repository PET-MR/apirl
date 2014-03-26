/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-raw.c                                                       *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : read (interactive) and write raw images                  *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * Functions    : MdcReadRAW()          - Read raw images interactive      *
 *                MdcWriteRAW()         - Write raw images to file         * 
 *                MdcInitRawPrevInput() - Initialize previous inputs       *
 *                MdcReadPredef()       - Read  predefined RAW settings    *
 *                MdcWritePredef()      - Write predefined RAW settings    *
 *                                                                         *
 * Notes        : Reading is an interactive process to determine           *
 *                the headersize to skip and the pixel data type           *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-raw.c,v 1.43 2010/08/28 23:44:23 enlf Exp $
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

/****************************************************************************
                              H E A D E R S
****************************************************************************/

#include "m-depend.h"

#include <stdio.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_STRINGS_H
#ifndef _WIN32
#include <strings.h>
#endif
#endif

#include "medcon.h"

/****************************************************************************
                              D E F I N E S 
****************************************************************************/

#define MDC_PREDEFSIG "# RPI v0.1"     /* predef signature    */

MdcRawInputStruct mdcrawinput;
MdcRawPrevInputStruct mdcrawprevinput;

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/
void MdcInitRawPrevInput(void)
{
   MdcRawPrevInputStruct *prev = &mdcrawprevinput;

   prev->XDIM=0; prev->YDIM=0; prev->NRIMGS=1;
   prev->GENHDR=0; prev->IMGHDR=0; prev->ABSHDR=0;
   prev->PTYPE = BIT16_S;
   prev->DIFF  = MDC_NO; prev->HDRREP=MDC_NO; prev->PSWAP=MDC_NO;
}


/* read raw images */
char *MdcReadRAW(FILEINFO *fi)
{
  MdcRawInputStruct *input = &mdcrawinput;
  MdcRawPrevInputStruct *prev = &mdcrawprevinput;
  IMG_DATA *id=NULL;
  Uint32 i, p, bytes, number;
  int AGAIN;
  double *pix=NULL;
  char *err=NULL;


  if (MDC_FILE_STDIN == MDC_YES)
    return("RAW  File read from stdin not possible");

  if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_BEGIN,0.,"Reading RAW:");

  /* put some defaults we use */
  fi->map = MDC_MAP_GRAY;

  if (XMDC_GUI == MDC_NO) 
  do { /* ask command-line info */
    input->gen_offset=0;
    input->img_offset=0;
    input->REPEAT=MDC_NO;
    input->DIFF=MDC_NO;
    input->abs_offset=NULL;
    AGAIN=MDC_NO;

    MDC_FILE_ENDIAN = MDC_HOST_ENDIAN;
 
    MdcPrintLine('-',MDC_FULL_LENGTH);
    MdcPrntScrn("\tINTERACTIVE  PROCEDURE\n");
    MdcPrintLine('-',MDC_FULL_LENGTH);

    number = prev->NRIMGS;

    MdcPrntScrn("\n\tFilename: %s\n\n",fi->ifname);
    MdcPrntScrn("\tNumber of images [%u]? ",number);
    if (!MdcPutDefault(mdcbufr)) 
      number = (Uint32)atol(mdcbufr); prev->NRIMGS=number;
    if (number == 0) return("RAW  No images specified");

    if (!MdcGetStructID(fi,number))
      return("RAW  Bad malloc IMG_DATA structs");

    MdcPrntScrn("\tGeneral header offset to binary data [%u bytes]? "
                  ,prev->GENHDR);
      if (MdcPutDefault(mdcbufr)) input->gen_offset = prev->GENHDR; 
      else{
        input->gen_offset = (Uint32)atol(mdcbufr);
        prev->GENHDR = input->gen_offset;
      }
    MdcPrntScrn("\tImage   header offset to binary data [%u bytes]? "
                  ,prev->IMGHDR);
      if (MdcPutDefault(mdcbufr)) input->img_offset = prev->IMGHDR;
      else{
        input->img_offset = (Uint32)atol(mdcbufr);
        prev->IMGHDR = input->img_offset;
      }
    MdcPrntScrn("\tImage   header repeated before each image ");
    sprintf(mdcbufr,"%s",MdcGetStrYesNo(prev->HDRREP));
    MdcPrntScrn("[%s]? ",mdcbufr); 
      if (!MdcPutDefault(mdcbufr)) { 
        if (mdcbufr[0]=='y' || mdcbufr[0]=='Y') {
          input->REPEAT = MDC_YES; prev->HDRREP = MDC_YES;
        }else{
          input->REPEAT = MDC_NO;  prev->HDRREP = MDC_NO;
        }
      }else{
         input->REPEAT = prev->HDRREP;
      }
      
    MdcPrntScrn("\tSwap the pixel bytes ");
    sprintf(mdcbufr,"%s",MdcGetStrYesNo(prev->PSWAP));
    MdcPrntScrn("[%s]? ",mdcbufr);
      if (!MdcPutDefault(mdcbufr)) {
        if (mdcbufr[0]=='y' || mdcbufr[0]=='Y') {
          MDC_FILE_ENDIAN = !MDC_HOST_ENDIAN; prev->PSWAP = MDC_YES;
         }else{
          MDC_FILE_ENDIAN = MDC_HOST_ENDIAN;  prev->PSWAP = MDC_NO; 
         }
      }else{
        if (prev->PSWAP == MDC_YES) {
          MDC_FILE_ENDIAN = !MDC_HOST_ENDIAN;
        }else{
          MDC_FILE_ENDIAN = MDC_HOST_ENDIAN;
        }
      }
          
    MdcPrntScrn("\tSame characteristics for all images ");
    sprintf(mdcbufr,"%s",MdcGetStrYesNo(!prev->DIFF));
    MdcPrntScrn("[%s]? ",mdcbufr);
      if (!MdcPutDefault(mdcbufr)) {
        if (mdcbufr[0]=='n' || mdcbufr[0]=='N') {
          input->DIFF=MDC_YES; prev->DIFF = MDC_YES;
        }else{
          input->DIFF=MDC_NO;  prev->DIFF = MDC_NO;
        }
      }else{
        input->DIFF = prev->DIFF;
      }

    /* malloc abs_offset[] */
    input->abs_offset=(Uint32 *)malloc(fi->number*sizeof(Uint32));
    if ( input->abs_offset == NULL ) {
      return("Couldn't malloc abs_offset[]");
    }else{
      memset(input->abs_offset,0,fi->number*sizeof(Uint32));
    } 
   
 
    for (i=0; i<fi->number; i++) {

       if (MDC_PROGRESS)
         MdcProgress(MDC_PROGRESS_INCR,1./(float)fi->number,NULL);
 
       id = &fi->image[i];

       if (input->DIFF) {
         MdcPrntScrn("\n\tIMAGE #%03u\n",i+1);
         MdcPrntScrn("\t----------\n");
       }else if (i==0) {
         MdcPrntScrn("\n\tALL IMAGES\n"); 
         MdcPrntScrn("\t----------\n");
       }

       /* put default */
       if (i==0) id->type = prev->PTYPE;

       if (input->DIFF || (!input->DIFF && i==0)) {

         MdcPrntScrn("\tAbsolute offset in bytes [%u]? ",prev->ABSHDR);
           if (MdcPutDefault(mdcbufr)) input->abs_offset[i] = prev->ABSHDR;
           else{ 
             input->abs_offset[i] = (Uint32)atol(mdcbufr);
             prev->ABSHDR = input->abs_offset[i];
           }
         MdcPrntScrn("\tImage columns [%u]? ",prev->XDIM); 
           if (MdcPutDefault(mdcbufr)) id->width = prev->XDIM;
           else{ id->width = (Uint32)atol(mdcbufr); prev->XDIM = id->width; }
           
           if(id->width == 0) return("RAW  No width specified");
         MdcPrntScrn("\tImage rows    [%u]? ",prev->YDIM); 
           if (MdcPutDefault(mdcbufr)) id->height = prev->YDIM;
           else{ id->height = (Uint32)atol(mdcbufr); prev->YDIM = id->height; }
           if (id->height == 0) return("RAW  No height specified");
         MdcPrntScrn("\tPixel data type:\n\n");
         MdcPrntScrn("\t\t %2d  ->  bit\n",BIT1);
         MdcPrntScrn("\t\t %2d  ->  Int8 \t\t %2d -> Uint8\n",BIT8_S,BIT8_U);
         MdcPrntScrn("\t\t %2d  ->  Int16\t\t %2d -> Uint16\n",BIT16_S,BIT16_U);
         MdcPrntScrn("\t\t %2d  ->  Int32\t\t %2d -> Uint32\n",BIT32_S,BIT32_U);
#ifdef HAVE_8BYTE_INT
         MdcPrntScrn("\t\t %2d  ->  Int64\t\t %2d -> Uint64\n",BIT64_S,BIT64_U);
#endif
         MdcPrntScrn("\t\t %2d  ->  float\t\t %2d -> double\n",FLT32,FLT64);
         MdcPrntScrn("\t\t %2d  ->  ascii\n",ASCII);
         MdcPrntScrn("\t\t %2d  ->  RGB\n\n",COLRGB);
         MdcPrntScrn("\tYour choice [%hu]? ", prev->PTYPE);

         if (MdcPutDefault(mdcbufr)) id->type = prev->PTYPE;
         else{ id->type = (Int16)atoi(mdcbufr); prev->PTYPE = id->type; }
         MdcPrntScrn("\n"); 
       }else{
         id->width = prev->XDIM;
         id->height = prev->YDIM; 
         id->type  = prev->PTYPE;
         input->abs_offset[i]= prev->ABSHDR;
       }
 
       switch (id->type) {
         case  BIT1   : 
         case  BIT8_S :
         case  BIT8_U :
         case  BIT16_S:
         case  BIT16_U:
         case  BIT32_S:
         case  BIT32_U:
#ifdef HAVE_8BYTE_INT
         case  BIT64_S:
         case  BIT64_U:
#endif
         case  FLT32  :
         case  FLT64  : 
         case  ASCII  : 
         case  COLRGB : id->bits = MdcType2Bits(id->type); break;
         default      : return("RAW  Unsupported data type");
       }
     }

     fi->endian = MDC_FILE_ENDIAN;
     fi->dim[0] = 3;
     fi->dim[3] = fi->number;

     MdcPrintImageLayout(fi,input->gen_offset,input->img_offset,
                            input->abs_offset,input->REPEAT);

     MdcPrntScrn("\n\tRedo input [no]? "); mdcbufr[0]='n'; AGAIN=MDC_NO;
      if (!MdcPutDefault(mdcbufr))
        if (mdcbufr[0]=='y' || mdcbufr[0]=='Y') {
          AGAIN=MDC_YES;
          MdcFreeIDs(fi);
          MdcFree(input->abs_offset);
        }

  }while ( AGAIN );

  if (MDC_VERBOSE) MdcPrntMesg("RAW  Reading <%s> ...",fi->ifname);

  fseek(fi->ifp,(signed)input->gen_offset,SEEK_SET);
 
  /* read the images */
  for (i = 0; i<fi->number; i++) {

     if ( i==0  || input->REPEAT)
       fseek(fi->ifp,(signed)input->img_offset,SEEK_CUR);

     if (input->abs_offset[i] != 0)
       fseek(fi->ifp,(signed)input->abs_offset[i],SEEK_SET);

     id = &fi->image[i];
     bytes = id->width * id->height * MdcType2Bytes(id->type);
     id->buf = MdcGetImgBuffer(bytes);
     if (id->buf == NULL) {
       MdcFree(input->abs_offset);
       return("RAW  Bad malloc image buffer");
     }
     if (id->type == ASCII) {
       pix = (double *)id->buf;
       for (p=0; p < (id->width * id->height); p++) {
          fscanf(fi->ifp,"%le",&pix[p]);
          if (ferror(fi->ifp)) {
            err=MdcHandleTruncated(fi,i+1,MDC_YES);
            if (err != NULL) {
              MdcFree(input->abs_offset);
              return(err);
            }
            break;
          }
       }
       id->type = FLT64; /* read ascii as double */
     }else{
      if (fread(id->buf,1,bytes,fi->ifp) != bytes) {
        err=MdcHandleTruncated(fi,i+1,MDC_YES); 
        if (err != NULL) {
          MdcFree(input->abs_offset);
          return(err);
        }
      }
     } 
     if (id->type == BIT1)  {
       MdcMakeBIT8_U(id->buf, fi, i);
       id->type = BIT8_U;
       id->bits = MdcType2Bits(id->type);
       if (i==0) { fi->type = id->type; fi->bits = id->bits; } 
     }

     if (id->type == COLRGB) fi->map = MDC_MAP_PRESENT; /* color */

     if (fi->truncated) break;
  }

  MdcFree(input->abs_offset);

  MdcCloseFile(fi->ifp);

  if (fi->truncated) return("RAW  Truncated image file");

  return NULL;
}

char *MdcWriteRAW(FILEINFO *fi)
{
  IMG_DATA *id;
  Uint32 size, i, p, bytes;
  Uint8 *new_buf=NULL, *pbuf=NULL;

  MDC_FILE_ENDIAN = MDC_WRITE_ENDIAN;


  /* print fileinfo to stderr */
  if (MDC_FILE_STDOUT == MDC_YES)  MdcPrintFI(fi);

  switch (fi->rawconv) {
   case MDC_FRMT_RAW:   
       if (XMDC_GUI == MDC_NO) 
         MdcDefaultName(fi,MDC_FRMT_RAW,fi->ofname,fi->ifname);
       break;
   case MDC_FRMT_ASCII: 
       if (XMDC_GUI == MDC_NO)
         MdcDefaultName(fi,MDC_FRMT_ASCII,fi->ofname,fi->ifname);
       break;
   default: return("Internal ## Improper `fi->rawconv' value");
  }

  if (MDC_PROGRESS) {
    switch (fi->rawconv) {
     case MDC_FRMT_RAW  : MdcProgress(MDC_PROGRESS_BEGIN,0.,"Writing RAW:");
         break;
     case MDC_FRMT_ASCII: MdcProgress(MDC_PROGRESS_BEGIN,0.,"Writing ASCII:");
         break;
    } 
  }

  if (MDC_VERBOSE) MdcPrntMesg("RAW  Writing <%s> ...",fi->ofname);

  /* indexed color no use without colormap */
  if ((fi->map == MDC_MAP_PRESENT) && (fi->type != COLRGB))
    return("RAW  Indexed colored files unsupported");

  if (MDC_FILE_STDOUT == MDC_YES) {
    fi->ofp = stdout;
  }else{
    if (MdcKeepFile(fi->ofname))
      return("RAW  File exists!!");
    if ( (fi->ofp=fopen(fi->ofname,"wb")) == NULL )
      return("RAW  Couldn't open file");
  }


  /* check some supported things */
  if (fi->type != COLRGB) {
    if (MDC_FORCE_INT != MDC_NO) {
      /* Sorry, no message.  The user should know ... */ 
    }else if (MDC_QUANTIFY || MDC_CALIBRATE) {
      if (fi->rawconv == MDC_FRMT_RAW) {
        MdcPrntWarn("RAW  Quantification to `float' type");
      }
    }
  }

  for (i=0; i<fi->number; i++) {

     if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_INCR,1./(float)fi->number,NULL);

     id = &fi->image[i];
     size = id->width * id->height;

     if (id->type == COLRGB) {               /* rgb */
       bytes = MdcType2Bytes(id->type);
       if (fwrite(id->buf,bytes,size,fi->ofp) != size) {
         return("RAW  Bad write RGB image");
       }
     }else if (MDC_FORCE_INT != MDC_NO) {    /* int */
       switch (MDC_FORCE_INT) {
         case BIT8_U : 
           new_buf=MdcGetImgBIT8_U(fi,i);
           if (new_buf == NULL) return("RAW  Bad malloc Uint8 buffer");
           break;
         case BIT16_S:
           new_buf=MdcGetImgBIT16_S(fi,i);
           if (new_buf == NULL) return("RAW  Bad malloc Int16 buffer");
           break;
         default:
           new_buf=MdcGetImgBIT16_S(fi,i);
           if (new_buf == NULL) return("RAW  Bad malloc Int16 buffer");
       }
       bytes = MdcType2Bytes(MDC_FORCE_INT);
       switch (fi->rawconv) {
         case MDC_FRMT_RAW:
           if (MDC_FILE_ENDIAN != MDC_HOST_ENDIAN)
             MdcMakeImgSwapped(new_buf,fi,i,id->width,id->height,MDC_FORCE_INT);
           if (fwrite(new_buf,bytes,size,fi->ofp) != size) {
             MdcFree(new_buf);
             return("RAW  Bad write integer image");
           }
           break;
         case MDC_FRMT_ASCII:
           for (pbuf=new_buf, p=0; p < size; p++, pbuf+=bytes) {
              MdcPrintValue(fi->ofp,pbuf,MDC_FORCE_INT); fprintf(fi->ofp," ");
              if ( ((p+1) % id->width) == 0 ) fprintf(fi->ofp,MDC_NEWLINE);
           }
           fprintf(fi->ofp,MDC_NEWLINE);
           break;
       }
     }else if (MDC_QUANTIFY || MDC_CALIBRATE) {
       new_buf=MdcGetImgFLT32(fi,i);
       if (new_buf == NULL) return("RAW  Quantification failed!");
       bytes = MdcType2Bytes(FLT32);
       switch (fi->rawconv) {
         case MDC_FRMT_RAW:
           if (MDC_FILE_ENDIAN != MDC_HOST_ENDIAN) 
             MdcMakeImgSwapped(new_buf,fi,i,id->width,id->height,FLT32);
           if (fwrite(new_buf,bytes,size,fi->ofp) != size) {
             MdcFree(new_buf);
             return("RAW  Bad write quantified image");
           }
           break;
         case MDC_FRMT_ASCII:
           for (pbuf = new_buf, p=0; p < size; p++, pbuf+=bytes) {
              MdcPrintValue(fi->ofp,pbuf,FLT32); fprintf(fi->ofp," ");
              if ( ((p+1) % id->width) == 0 ) fprintf(fi->ofp,MDC_NEWLINE);
           }
           fprintf(fi->ofp,MDC_NEWLINE);
           break;
       }
     }else{ /* same pixel type */
       bytes = MdcType2Bytes(id->type);
       switch (fi->rawconv) {
         case MDC_FRMT_RAW:
           if (MDC_FILE_ENDIAN != MDC_HOST_ENDIAN) {
             new_buf = MdcGetImgSwapped(fi,i);
             if (fwrite(new_buf,bytes,size,fi->ofp) != size) {
               MdcFree(new_buf);
               return("RAW  Bad write swapped image");
             }
           }else if (fwrite(id->buf,bytes,size,fi->ofp) != size) {
             return("RAW  Bad write original image ");
           }
           break;
         case MDC_FRMT_ASCII:
           for (pbuf=id->buf, p=0; p < size; p++, pbuf+=bytes) {
              MdcPrintValue(fi->ofp,pbuf,id->type); fprintf(fi->ofp," ");
              if ( ((p+1) % id->width) == 0 ) fprintf(fi->ofp,MDC_NEWLINE);
           }
           fprintf(fi->ofp,MDC_NEWLINE);
           break;
       }
     }

     MdcFree(new_buf); /* free when allocated */
  }

  MdcCloseFile(fi->ofp);
 
  return NULL;

}

int MdcCheckPredef(const char *fname)
{
  FILE *fp;
  char sig[10];

  if ((fp = fopen(fname,"rb")) == NULL) return(MDC_NO);

  fread(sig,1,10,fp); MdcCloseFile(fp);

  if ( memcmp(sig,MDC_PREDEFSIG,10) ) return(MDC_NO);

  return(MDC_YES);
}

char *MdcReadPredef(const char *fname)
{
  MdcRawPrevInputStruct *prev = &mdcrawprevinput;
  FILE *fp;

  prev->DIFF = MDC_NO;
  prev->PSWAP = MDC_NO;
  prev->HDRREP = MDC_NO;
 
  if ((fp = fopen(fname,"rb")) == NULL) {
    return("Couldn't open raw predef input file");
  }else{
    MdcGetStrLine(mdcbufr,80,fp); prev->NRIMGS=(Uint32)atoi(mdcbufr);
    MdcGetStrLine(mdcbufr,80,fp); prev->GENHDR=(Uint32)atoi(mdcbufr);
    MdcGetStrLine(mdcbufr,80,fp); prev->IMGHDR=(Uint32)atoi(mdcbufr);
    MdcGetStrLine(mdcbufr,80,fp); if (mdcbufr[0] == 'y') prev->HDRREP = MDC_YES;
    MdcGetStrLine(mdcbufr,80,fp); if (mdcbufr[0] == 'y') prev->PSWAP = MDC_YES;
    MdcGetStrLine(mdcbufr,80,fp); if (mdcbufr[0] == 'y') { } /*no DIFF allowed*/
    MdcGetStrLine(mdcbufr,80,fp); prev->ABSHDR=(Uint32)atoi(mdcbufr);
    MdcGetStrLine(mdcbufr,80,fp); prev->XDIM=(Uint32)atoi(mdcbufr);
    MdcGetStrLine(mdcbufr,80,fp); prev->YDIM=(Uint32)atoi(mdcbufr);
    MdcGetStrLine(mdcbufr,80,fp); prev->PTYPE=(Int16)atoi(mdcbufr);
    /* MdcGetStrLine(mdcbufr,80,fp); */  /* redo */
  }

  if (ferror(fp)) {
    MdcCloseFile(fp); return("Error reading raw predef input file");
  }

  MdcCloseFile(fp);

  return(NULL);

}

/* write predefined RAW settings, suitable */
/* as input for interactive read           */
char *MdcWritePredef(const char *fname)
{
  FILE *fp;
  MdcRawPrevInputStruct *prev = &mdcrawprevinput;

  if (MdcKeepFile(fname))
    return("Raw predef input file already exists!!");
 
  if ((fp = fopen(fname,"w")) == NULL) {
    return("Couldn't open writeable raw predef input file");
  }else{
    fprintf(fp,"%s - BEGIN #\n#\n",MDC_PREDEFSIG); /* MDC_PREDEFSIG - BEGIN */
    fprintf(fp,"# Total number of images?\n%u\n",prev->NRIMGS);
    fprintf(fp,"# General header offset (bytes)?\n%u\n",prev->GENHDR);
    fprintf(fp,"# Image   header offset (bytes)?\n%u\n",prev->IMGHDR);
    fprintf(fp,"# Repeated image header?\n");
    if (prev->HDRREP == MDC_YES) {
      fprintf(fp,"yes\n");
    }else{
      fprintf(fp,"no\n");
    }
    fprintf(fp,"# Swap pixel bytes?\n");
    if (prev->PSWAP == MDC_YES) {
      fprintf(fp,"yes\n");
    }else{
      fprintf(fp,"no\n");
    }
    fprintf(fp,"# Identical images?\nyes\n");
    fprintf(fp,"# Absolute offset in bytes?\n%u\n",prev->ABSHDR);
    fprintf(fp,"# Image columns?\n%u\n",prev->XDIM);
    fprintf(fp,"# Image rows?\n%u\n",prev->YDIM);
    fprintf(fp,"# Pixel data type?\n%hu\n",prev->PTYPE);
    fprintf(fp,"# Redo input?\nno\n");
    fprintf(fp,"#\n%s - END #\n",MDC_PREDEFSIG);
  }

  if (ferror(fp)) {
    MdcCloseFile(fp); return("Failure to write raw predef input file");
  }

  MdcCloseFile(fp);

  return(NULL);
}


