/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mindspore.styletransferdemo;

import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.net.Uri;
import android.provider.MediaStore;

import androidx.annotation.NonNull;
import androidx.exifinterface.media.ExifInterface;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ImageUtils {

    private static final String TAG = "ImageUtils";
    private static final int FULL_QUALITY = 100;

    private static Matrix decodeExifOrientation(int orientation) {
        Matrix matrix = new Matrix();

        switch (orientation) {
            case ExifInterface.ORIENTATION_NORMAL:
            case ExifInterface.ORIENTATION_UNDEFINED:
                break;
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.postRotate(90F);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.postRotate(180F);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.postRotate(270F);
                break;
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                matrix.postScale(-1F, 1F);
                break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                matrix.postScale(1F, -1F);
                break;
            case ExifInterface.ORIENTATION_TRANSPOSE:
                matrix.postScale(-1F, 1F);
                matrix.postRotate(270F);
                break;
            case ExifInterface.ORIENTATION_TRANSVERSE:
                matrix.postScale(-1F, 1F);
                matrix.postRotate(90F);
                break;

            default:
                try {
                    new IllegalArgumentException("Invalid orientation: " + orientation);
                } catch (Throwable throwable) {
                    throwable.printStackTrace();
                }
        }
        return matrix;
    }

    public void setExifOrientation(@NonNull String filePath, @NonNull String value) {
        try {
            ExifInterface exif = new ExifInterface(filePath);
            exif.setAttribute(ExifInterface.TAG_ORIENTATION, value);
            exif.saveAttributes();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public int computeExifOrientation(int rotationDegrees, boolean mirrored) {
        if (rotationDegrees == 0 && !mirrored) {
            return ExifInterface.ORIENTATION_NORMAL;
        } else if (rotationDegrees == 0 && mirrored) {
            return ExifInterface.ORIENTATION_FLIP_HORIZONTAL;
        } else if (rotationDegrees == 180 && !mirrored) {
            return ExifInterface.ORIENTATION_ROTATE_180;
        } else if (rotationDegrees == 180 && mirrored) {
            return ExifInterface.ORIENTATION_FLIP_VERTICAL;
        } else if (rotationDegrees == 90 && !mirrored) {
            return ExifInterface.ORIENTATION_ROTATE_90;
        } else if (rotationDegrees == 90 && mirrored) {
            return ExifInterface.ORIENTATION_TRANSPOSE;
        } else if (rotationDegrees == 270 && !mirrored) {
            return ExifInterface.ORIENTATION_ROTATE_270;
        } else if (rotationDegrees == 270 && mirrored) {
            return ExifInterface.ORIENTATION_TRANSVERSE;
        } else {
            return ExifInterface.ORIENTATION_UNDEFINED;
        }
    }

    public static Bitmap decodeBitmap(@NonNull File file) {
        Bitmap finalBitmap = null;
        try {
            ExifInterface exif = new ExifInterface(file.getAbsolutePath());

            Matrix transformation = decodeExifOrientation(exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_ROTATE_90));

            BitmapFactory.Options options = new BitmapFactory.Options();
            Bitmap bitmap = BitmapFactory.decodeFile(file.getAbsolutePath(), options);
            finalBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), transformation, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return finalBitmap;
    }


    public static Bitmap scaleBitmapAndKeepRatio(Bitmap targetBmp, int reqHeightInPixels, int reqWidthInPixels) {
        if (targetBmp.getHeight() == reqHeightInPixels && targetBmp.getWidth() == reqWidthInPixels) {
            return targetBmp;
        }

        Matrix matrix = new Matrix();
        matrix.setRectToRect(new RectF(0f, 0f,
                targetBmp.getWidth(),
                targetBmp.getHeight()
        ), new RectF(0f, 0f,
                reqWidthInPixels,
                reqHeightInPixels
        ), Matrix.ScaleToFit.FILL);

        return Bitmap.createBitmap(
                targetBmp, 0, 0,
                targetBmp.getWidth(),
                targetBmp.getHeight(), matrix, true
        );
    }

    public static Bitmap loadBitmapFromResources(Context context, String path) {
        try {
            InputStream inputStream = context.getAssets().open(path);
            return BitmapFactory.decodeStream(inputStream);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static ByteBuffer bitmapToByteBuffer(Bitmap bitmapIn, int width, int height, float mean, float std) {
        Bitmap bitmap = scaleBitmapAndKeepRatio(bitmapIn, width, height);
        ByteBuffer inputImage = ByteBuffer.allocateDirect(1 * width * height * 3 * 4);
        inputImage.order(ByteOrder.nativeOrder());
        inputImage.rewind();
        int[] intValues = new int[width * height];
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height);
        int pixel = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int value = intValues[pixel++];
                inputImage.putFloat(((float) (value >> 16 & 255) - mean) / std);
                inputImage.putFloat(((float) (value >> 8 & 255) - mean) / std);
                inputImage.putFloat(((float) (value & 255) - mean) / std);
            }
        }
        inputImage.rewind();
        return inputImage;
    }


    public static Bitmap convertArrayToBitmap(float[][][][] imageArray, int imageWidth, int imageHeight) {
        Bitmap styledImage = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);

        for (int x = 0; x < imageArray[0].length; x++) {
            for (int y = 0; y < imageArray[0][0].length; y++) {

                int color = Color.rgb((int) (imageArray[0][x][y][0] * (float) 255),
                        (int) (imageArray[0][x][y][1] * (float) 255),
                        (int) (imageArray[0][x][y][2] * (float) 255));
                // this y, x is in the correct order!!!
                styledImage.setPixel(y, x, color);
            }
        }
        return styledImage;
    }

    public Bitmap createEmptyBitmap(int imageWidth, int imageHeigth, int color) {
        Bitmap ret = Bitmap.createBitmap(imageWidth, imageHeigth, Bitmap.Config.RGB_565);
        if (color != 0) {
            ret.eraseColor(color);
        }
        return ret;
    }

    // Save the picture to the system album and refresh it.
    public static void saveToAlbum(final Context context, Bitmap bitmap) {
        Uri saveUri = context.getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, new ContentValues());
        OutputStream outputStream = null;
        try {
            outputStream = context.getContentResolver().openOutputStream(saveUri);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        if (bitmap.compress(Bitmap.CompressFormat.JPEG, FULL_QUALITY, outputStream)) {
            Intent intent = new Intent();
            intent.setData(saveUri);
            context.sendBroadcast(intent);
            context.sendBroadcast(new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE, saveUri));
        }
    }
}