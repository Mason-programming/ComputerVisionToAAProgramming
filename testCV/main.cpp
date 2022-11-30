#include "StartCamera.h"


int main(int argc, const char * argv[]) {
    
    int choice;
    cout << "1. Recoginze face" << endl;
    cout << "2. Add Face." << endl;
    cout << "3. Train model" << endl;
    cout << "4. Add eye" << endl;
    cin >> choice;
    
    switch(choice){
        case 1:
            faceRecognizer();
            break;
        case 2:
            addFace();
            break;
        case 3:
            eigenFaceTrainer();
            break;
        case 4:
            detectBothEyes();
            break;
        default:
            return 0;
    }
    return 0;
}
