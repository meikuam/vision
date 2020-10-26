package org.pytorch.testapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.TextureView;
import android.view.View;
import android.view.ViewStub;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.annotation.WorkerThread;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.core.app.ActivityCompat;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class CameraActivity extends AppCompatActivity {
  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    NativeLoader.loadLibrary("pytorch_jni");
    NativeLoader.loadLibrary("torchvision");
  }

  private static final String TAG = BuildConfig.LOGCAT_TAG;
  private static final int TEXT_TRIM_SIZE = 4096;

  private static final int REQUEST_CODE_CAMERA_PERMISSION = 200;
  private static final String[] PERMISSIONS = {Manifest.permission.CAMERA};

  private long mLastAnalysisResultTime;

  protected HandlerThread mBackgroundThread;
  protected Handler mBackgroundHandler;
  protected Handler mUIHandler;

  private TextView mTextView;
  private ImageView mCameraOverlay;
  private StringBuilder mTextViewStringBuilder = new StringBuilder();

  private Paint mPaint;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_camera);
    mTextView = findViewById(R.id.text);
    mCameraOverlay = findViewById(R.id.camera_overlay);
    mUIHandler = new Handler(getMainLooper());
    startBackgroundThread();

    if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
        != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(this, PERMISSIONS, REQUEST_CODE_CAMERA_PERMISSION);
    } else {
      setupCameraX();
    }
    mPaint = new Paint();
    mPaint.setAntiAlias(true);
    mPaint.setDither(true);
    mPaint.setColor(Color.GREEN);
  }

  @Override
  protected void onPostCreate(@Nullable Bundle savedInstanceState) {
    super.onPostCreate(savedInstanceState);
    startBackgroundThread();
  }

  protected void startBackgroundThread() {
    mBackgroundThread = new HandlerThread("ModuleActivity");
    mBackgroundThread.start();
    mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
  }

  @Override
  protected void onDestroy() {
    stopBackgroundThread();
    super.onDestroy();
  }

  protected void stopBackgroundThread() {
    mBackgroundThread.quitSafely();
    try {
      mBackgroundThread.join();
      mBackgroundThread = null;
      mBackgroundHandler = null;
    } catch (InterruptedException e) {
      Log.e(TAG, "Error on stopping background thread", e);
    }
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
      if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
        Toast.makeText(
            this,
            "You can't use image classification example without granting CAMERA permission",
            Toast.LENGTH_LONG)
            .show();
        finish();
      } else {
        setupCameraX();
      }
    }
  }

  private static final int TENSOR_WIDTH = 96;
  private static final int TENSOR_HEIGHT = 96;

  private void setupCameraX() {
    final TextureView textureView =
        ((ViewStub) findViewById(R.id.camera_texture_view_stub))
            .inflate()
            .findViewById(R.id.texture_view);
    final PreviewConfig previewConfig = new PreviewConfig.Builder().build();
    final Preview preview = new Preview(previewConfig);
    preview.setOnPreviewOutputUpdateListener(
        new Preview.OnPreviewOutputUpdateListener() {
          @Override
          public void onUpdated(Preview.PreviewOutput output) {
            textureView.setSurfaceTexture(output.getSurfaceTexture());
          }
        });

    final ImageAnalysisConfig imageAnalysisConfig =
        new ImageAnalysisConfig.Builder()
            .setTargetResolution(new Size(TENSOR_WIDTH, TENSOR_HEIGHT))
            .setCallbackHandler(mBackgroundHandler)
            .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
            .build();
    final ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);
    imageAnalysis.setAnalyzer(
        new ImageAnalysis.Analyzer() {
          @Override
          public void analyze(ImageProxy image, int rotationDegrees) {
            if (SystemClock.elapsedRealtime() - mLastAnalysisResultTime < 500) {
              return;
            }

            final Result result = CameraActivity.this.analyzeImage(image, rotationDegrees);

            if (result != null) {
              mLastAnalysisResultTime = SystemClock.elapsedRealtime();
              CameraActivity.this.runOnUiThread(
                  new Runnable() {
                    @Override
                    public void run() {
                      CameraActivity.this.handleResult(result);
                    }
                  });
            }
          }
        });

    CameraX.bindToLifecycle(this, preview, imageAnalysis);
  }

  private Module mModule;
  private FloatBuffer mInputTensorBuffer;
  private Tensor mInputTensor;

  private static final float[] NO_NORM_MEAN = {0.f, 0.f, 0.f};
  private static final float[] NO_NORM_STD = {1.f, 1.f, 1.f};

  private static class Box {
    public final float score;
    public final float x0;
    public final float y0;
    public final float x1;
    public final float y1;

    public Box(float score, float x0, float y0, float x1, float y1) {
      this.score = score;
      this.x0 = x0;
      this.y0 = y0;
      this.x1 = x1;
      this.y1 = y1;
    }
  }

  private final List<Box> mBoxes = new ArrayList<>(4);

  @WorkerThread
  @Nullable
  protected Result analyzeImage(ImageProxy image, int rotationDegrees) {
    Log.i(TAG, String.format("analyzeImage(%s, %d)", image, rotationDegrees));
    if (mModule == null) {
      Log.i(TAG, "Loading module from asset '" + BuildConfig.MODULE_ASSET_NAME + "'");
      mInputTensorBuffer = Tensor.allocateFloatBuffer(3 * TENSOR_HEIGHT * TENSOR_WIDTH);
      mInputTensor = Tensor.fromBlob(mInputTensorBuffer, new long[]{3, TENSOR_HEIGHT, TENSOR_WIDTH});
      mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), BuildConfig.MODULE_ASSET_NAME);
    }
    mBoxes.clear();
    final long startTime = SystemClock.elapsedRealtime();
    TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
        image.getImage(),
        rotationDegrees,
        TENSOR_WIDTH,
        TENSOR_HEIGHT,
        NO_NORM_MEAN,
        NO_NORM_STD,
        mInputTensorBuffer,
        0);
    final long moduleForwardStartTime = SystemClock.elapsedRealtime();
    final IValue outputTuple = mModule.forward(IValue.listFrom(mInputTensor));
    final IValue out1 = outputTuple.toTuple()[1];
    final Map<String, IValue> map = out1.toList()[0].toDictStringKey();

    float[] boxesData = new float[]{};
    float[] scoresData = new float[]{};
    if (map.containsKey("boxes")) {
      final Tensor boxes = map.get("boxes").toTensor();
      final Tensor scores = map.get("scores").toTensor();
      boxesData = boxes.getDataAsFloatArray();
      scoresData = scores.getDataAsFloatArray();
      final int n = scoresData.length;


      for (int i = 0; i < n; i++) {
        Box box = new Box(
            scoresData[i],
            boxesData[4 * i + 0],
            boxesData[4 * i + 1],
            boxesData[4 * i + 2],
            boxesData[4 * i + 3]
        );
        android.util.Log.i(TAG,
            String.format("Forward result %d: score %f box:(%f, %f, %f, %f)",
                i, box.score, box.x0, box.y0, box.x1, box.y1));

        if (i < 4) {
          mBoxes.add(box);
        }
      }
    } else {
      android.util.Log.i(TAG, "Forward result empty");
    }

    final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;
    final long analysisDuration = SystemClock.elapsedRealtime() - startTime;
    return new Result(boxesData, scoresData, moduleForwardDuration, analysisDuration);
  }

  @UiThread
  protected void handleResult(Result result) {
    final int W = mCameraOverlay.getMeasuredWidth();
    final int H = mCameraOverlay.getMeasuredHeight();
    int offsetX = 0;
    int offsetY = 0;
    int size = 0;
    if (H > W) {
      offsetX = 0;
      size = H - W;
      offsetY = size / 2;
    } else {
      size = W - H;
      offsetX = size / 2;
      offsetY = 0;
    }
    float scaleX = (float) size / TENSOR_WIDTH;
    float scaleY = (float) size / TENSOR_HEIGHT;
    final Bitmap bitmap = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888);
    final Canvas canvas = new Canvas(bitmap);

    final int n = result.scores.length;
    final int nBoxes = Math.min(n, 100);
    for (int i = 0; i < nBoxes; i++) {
      float x0 = result.boxes[i];
      float y0 = result.boxes[i + 1];
      float x1 = result.boxes[i + 2];
      float y1 = result.boxes[i + 3];
      float c_x0 = offsetX + scaleX * x0;
      float c_y0 = offsetY + scaleY * y0;

      float c_x1 = offsetX + scaleX * x1;
      float c_y1 = offsetY + scaleY * y1;

      canvas.drawLine(c_x0, c_y0, c_x1, c_y0, mPaint);
      canvas.drawLine(c_x1, c_y0, c_x1, c_y1, mPaint);
      canvas.drawLine(c_x1, c_y1, c_x0, c_y1, mPaint);
      canvas.drawLine(c_x0, c_y1, c_x0, c_y0, mPaint);
      canvas.drawText(String.format("%.2f", result.scores[i]), c_x0, c_y0, mPaint);
    }
    mCameraOverlay.setImageBitmap(bitmap);

    String message = String.format("forwardDuration:%d", result.moduleForwardDuration);
    Log.i(TAG, message);
    mTextViewStringBuilder.insert(0, '\n').insert(0, message);
    if (mTextViewStringBuilder.length() > TEXT_TRIM_SIZE) {
      mTextViewStringBuilder.delete(TEXT_TRIM_SIZE, mTextViewStringBuilder.length());
    }
    mTextView.setText(mTextViewStringBuilder.toString());
  }
}
