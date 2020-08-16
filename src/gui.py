from   PyQt5.QtWidgets import ( QMainWindow, QApplication, QPushButton, QLineEdit, QLabel, QWidget,
                                QGridLayout, QFileDialog )
from   PyQt5.QtGui     import   QPixmap
from   predict         import   Prediction
import cv2

class MainWindow( QMainWindow ):
    """Class implementing GUI"""
    
    def __init__( self ):
        """Constructor."""
        
        super( QMainWindow, self ).__init__( )

        self.setGeometry(10, 50, 300, 150)
        self.setWindowTitle("Informative or NonInformative Tweets")

        self.filename   = ""
        self.prediction = Prediction( "../models/text.h5", "../models/old_image/damage.h5" )
        
        self.widget = QWidget    ( )
        self.widget.setObjectName("widget")
        self.widget.setStyleSheet("QWidget#widget{border-image:url('../images/background.jpg');}\n")
        
        self.layout = QGridLayout( self.widget )

        self.image_label  = QLabel( )
        self.result_label = QLabel( )
        
        self.image_label .setScaledContents( True )
        self.result_label.setScaledContents( True )
        
        self.browse     = QPushButton( "Browse" )
        self.text_field = QLineEdit  ( )
        self.submit     = QPushButton( "Submit" )

        self.layout.addWidget( self.image_label )
        self.layout.addWidget( self.browse )
        self.layout.addWidget( self.text_field )
        self.layout.addWidget( self.submit )
        self.layout.addWidget( self.result_label )

        self.browse.clicked.connect( self.pick_file )
        self.submit.clicked.connect( self.predict )

        self.setCentralWidget( self.widget )

    def pick_file( self ):
        """For selecting a file."""

        self.filename = QFileDialog.getOpenFileName( )[ 0 ]

        if self.filename != "":

            pixmap          = QPixmap ( self.filename )
            self.image_label.setPixmap( pixmap )

    def predict( self ):
        
        if self.filename != "":
            
            image               = cv2.imread( self.filename )
            text                = self.text_field.text( )
            im_class, txt_class = self.prediction.run( image, text )
            
            if im_class == 0 and txt_class == 0:
                self.result_label.setText( "<font color='red'>Informative</font>" )
            else:
                self.result_label.setText( "<font color='green'>Non-Informative</font>" )
            

if __name__=="__main__":
    app=QApplication( [ ] )
    mw=MainWindow   ( )
    mw.show         ( )
    app.exec_       ( )
    
        
