/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
package com.mindspore.common.net;

import io.reactivex.Observable;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Streaming;

public interface RetrofitService {

    @GET("newversion.json")
    Call<UpdateInfoBean> getUpdateInfo();

    @Streaming
    @GET("MindSpore_inhand.apk")
    Observable<ResponseBody> downloadApk();

    @Streaming
    @GET("danceFactory/dance_factory_demo2.mp4")
    Observable<ResponseBody> downloadDanceVideo();

}


