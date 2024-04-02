#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

VectorXd risolto_con_palu(MatrixXd A, VectorXd b) {
    PartialPivLU<MatrixXd> lu(A);
    VectorXd x = lu.solve(b);
    return x;
}

VectorXd risolto_con_qr(MatrixXd A, VectorXd b) {
    HouseholderQR<MatrixXd> qr(A);
    VectorXd x = qr.solve(b);
    return x;
}

double errore_relativo(VectorXd x, VectorXd x_true) {
    return (x - x_true).norm() / x_true.norm();
}

int main() {
    MatrixXd A1(2, 2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    VectorXd b1(2);
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    VectorXd x_true1(2);
    x_true1 << -1.0e+0, -1.0e+00;

    MatrixXd A2(2, 2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    VectorXd b2(2);
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    VectorXd x_true2(2);
    x_true2 << -1.0e+0, -1.0e+00;

    MatrixXd A3(2, 2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    VectorXd b3(2);
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    VectorXd x_true3(2);
    x_true3 << -1.0e+0, -1.0e+00;

    // Risolve i sistemi e calcola i relativi errori
    VectorXd x_palu, x_qr;
    double err_rel_palu, err_rel_qr;

    // Sistema 1
    x_palu = risolto_con_palu(A1, b1);
    err_rel_palu = errore_relativo(x_palu, x_true1);
    x_qr = risolto_con_qr(A1, b1);
    err_rel_qr = errore_relativo(x_qr, x_true1);
    cout << "Sistema 1:" << endl;
    cout << "Soluzione di decomposizione PALU: " << endl << x_palu << endl;
    cout << "Errore relativo con la decomposizione PALU: " << err_rel_palu << endl;
    cout << "Soluzione di decomposizione QR: " << endl << x_qr << endl;
    cout << "Errore relativo con la decomposizione QR: " << err_rel_qr << endl;

    // Sistema 2
    x_palu = risolto_con_palu(A2, b2);
    err_rel_palu = errore_relativo(x_palu, x_true2);
    x_qr = risolto_con_qr(A2, b2);
    err_rel_qr = errore_relativo(x_qr, x_true2);
    cout << "Sistema 2:" << endl;
    cout << "Soluzione di decomposizione PALU: " << endl << x_palu << endl;
    cout << "Errore relativo con la decomposizione PALU: " << err_rel_palu << endl;
    cout << "Soluzione di decomposizione QR: " << endl << x_qr << endl;
    cout << "Errore relativo con la decomposizione QR: " << err_rel_qr << endl;

    // Sistema 3
    x_palu = risolto_con_palu(A3, b3);
    err_rel_palu = errore_relativo(x_palu, x_true3);
    x_qr = risolto_con_qr(A3, b3);
    err_rel_qr = errore_relativo(x_qr, x_true3);
    cout << "Sistema 3:" << endl;
    cout << "Soluzione di decomposizione PALU: " << endl << x_palu << endl;
    cout << "Errore relativo con la decomposizione PALU: " << err_rel_palu << endl;
    cout << "Soluzione di decomposizione QR: " << endl << x_qr << endl;
    cout << "Errore relativo con la decomposizione QR: " << err_rel_qr << endl;

    return 0;
}
