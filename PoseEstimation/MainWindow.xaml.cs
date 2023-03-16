using Microsoft.UI.Xaml;
using System;
using System.IO;
using Microsoft.UI.Xaml.Media.Imaging;
using System.Threading;
using System.Drawing;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using OpenCvSharp.Dnn;
using System.Diagnostics;

namespace PoseEstimation {
    public sealed partial class MainWindow : Microsoft.UI.Xaml.Window
    {
        private readonly VideoCaptureAPIs _videoCaptureApi = VideoCaptureAPIs.DSHOW;
        private readonly VideoCapture _videoCapture;
        private Mat _capturedFrame = new Mat();

        private Thread _captureThread;
        private readonly ManualResetEventSlim _threadStopEvent = new ManualResetEventSlim(false);

        private Net net;
        private Mat out1, heatMap;

        private int frameWidth = 360, frameHeight=240;

        private OpenCvSharp.Point pm, p;
        private double minVal, maxVal;

        Brush brushEclipse = Brushes.LightGreen;


        private float[,] outputPoints=new float[20,3];

        private string[] BODY_PARTS = { "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
            "LShoulder", "LElbow", "LWrist", "RHip", "RKnee","RAnkle", "LHip",
            "LKnee", "LAnkle", "REye","LEye", "REar", "LEar", "Background" };
        public MainWindow() {
            this.InitializeComponent();
            Closed += MainWindow_Closed;

            _videoCapture = VideoCapture.FromCamera(0, _videoCaptureApi);

            _captureThread = new Thread(startCapture);
            _captureThread.IsBackground = true;                                 // Make the thread Background
            _captureThread.Start();

            net = CvDnn.ReadNetFromTensorflow("C:/graph_opt.pb");
        }

        private async void startCapture() {
            _videoCapture.Open(0, _videoCaptureApi);
            frameWidth = _videoCapture.FrameWidth;
            frameHeight = _videoCapture.FrameHeight;

            _videoCapture.Fps = 5;

            OpenCvSharp.Size s1 = new OpenCvSharp.Size(frameWidth, frameHeight);
            Scalar s2 = new Scalar(127.5, 127.5, 127.5);

            while (!_threadStopEvent.Wait(0)) {
                _videoCapture.Read(_capturedFrame);
                if (!(_capturedFrame.Empty())) {
                    _ = this.DispatcherQueue.TryEnqueue(() => {
                        Cv2.SetNumThreads(8);                                   //Limit number of threads to reduce CPU load
                        Bitmap bitmapFrameRotation = BitmapConverter.ToBitmap(_capturedFrame);

                        net.SetInput(CvDnn.BlobFromImage(_capturedFrame, 1.0, s1, s2, true, false));

                        out1 = net.Forward();
                        int H = out1.Size(2);
                        int W = out1.Size(3);
                        float SX = (float)frameWidth / W;
                        float SY = (float)frameHeight / H;

                        for (int i = 0; i <= 18; i++) {
                            heatMap = new Mat(H, W, MatType.CV_32F, out1.Ptr(0, i));
                            Cv2.MinMaxLoc(heatMap, out minVal, out maxVal, out p, out pm);
                            outputPoints[i,0] = pm.X* SX;
                            outputPoints[i, 1] = pm.Y * SY;

                            outputPoints[i, 2] = (float)Math.Round(maxVal,3);
                        }
                        //Debug.WriteLine(outputPoints[0,0]);

                        Graphics g = Graphics.FromImage(bitmapFrameRotation);

                        for (int i = 0; i <= 18; i++) {
                            if ((outputPoints[i, 0] > 0) && (outputPoints[i, 0] > 0)) {
                                if (outputPoints[i, 2] > 0.25) {
                                    brushEclipse= Brushes.LightGreen;
                                }
                                else if (outputPoints[i, 2] < 0.15) {
                                    brushEclipse = Brushes.Red;
                                }
                                else {
                                    brushEclipse = Brushes.Blue;
                                }
                                if (outputPoints[i, 2] > 0.15) {
                                    g.FillEllipse(brushEclipse, outputPoints[i, 0], outputPoints[i, 1], 10, 10);
                                    g.DrawString(BODY_PARTS[i] + " , " + outputPoints[i, 2].ToString(), new Font("Ariel", 10), brushEclipse, new PointF(outputPoints[i, 0]+10, outputPoints[i, 1]));
                                }

                            }
                        }

                        BitmapImage bitmapImage = new BitmapImage();
                        using (MemoryStream stream = new MemoryStream()) {
                            bitmapFrameRotation.Save(stream, System.Drawing.Imaging.ImageFormat.Png);
                            stream.Position = 0;
                            bitmapImage.SetSource(stream.AsRandomAccessStream());
                        }
                        imagePreview.Source = bitmapImage;
                    });
                }
            }
        }
        private void MainWindow_Closed(object sender, WindowEventArgs args) {
            Environment.Exit(Environment.ExitCode);                         // Close all thread
        }
    }
}
