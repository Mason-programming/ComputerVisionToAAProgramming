using System.Runtime.InteropServices;

namespace FaceRecognitionAPP.ComputerVision
{
    class StartCamera
    {
        public const string CppStartCameraDLL = @"\\10.2.63.252\testCV\FaceRecognitionAPP\bin\Debug\netcoreapp3.1\FaceRecognitionAPP.dll";

        [DllImport(CppStartCameraDLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern int TurnCameraOn();

        public void turnCameraOn()
        {
            TurnCameraOn();
        }
    }
}
