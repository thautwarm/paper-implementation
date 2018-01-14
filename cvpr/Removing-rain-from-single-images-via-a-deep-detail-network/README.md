## 去雨

- CPU

    ```shell
    python cpu_rain_removal.py
    ```
- GPU  

    如果使用gpu，我默认你的内存在16g以上，也就是不使用磁盘缓存训练数据。

    ```shell
    python gpu_rain_removal.py
    ```

- 测试(取消代码画图部分的注释即可显示比较图):

    ```shell
    python test_rain_removal.py
    ```

## 去噪

- GPU

    训练:
    ```shell
    python gpu_denoise.py
    ```

- 测试(取消代码画图部分的注释即可显示比较图):

    ```shell
    python test_denoise.py
    ```
