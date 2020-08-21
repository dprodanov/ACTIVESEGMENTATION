package activeSegmentation.gui;


import activeSegmentation.ASCommon;
import activeSegmentation.LearningType;
import activeSegmentation.ProjectType;
import activeSegmentation.feature.FeatureManager;
import activeSegmentation.prj.ProjectInfo;
import activeSegmentation.prj.ProjectManager;
import activeSegmentation.util.GuiUtil;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.ImageWindow;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.gui.TextRoi;
import ij.process.ImageProcessor;
import ij.process.LUT;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.*;

public class FeaturePanelNew extends ImageWindow implements ASCommon  {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private FeatureManager featureManager;
	/** opacity (in %) of the result overlay image */
	int overlayOpacity = 33;
	/** alpha composite for the result overlay image */
	Composite overlayAlpha = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, overlayOpacity / 100f);
	private ImageOverlay resultOverlay;
	LUT overlayLUT;
	/** flag to display the overlay image */
	private boolean showColorOverlay=false;
	ImagePlus classifiedImage;
	// Create overlay LUT
	byte[] red = new byte[ 256 ];
	byte[] green = new byte[ 256 ];
	byte[] blue = new byte[ 256 ];

	private Map<String, JList<String>> exampleList;
	private Map<String, JList<String>> allexampleList;

	/** array of ROI list overlays to paint the transparent ROIs of each class */
	private Map<String,RoiListOverlay> roiOverlayList;

	/** Used only during classification setting*/
	private Map<String,Integer> predictionResultClassification;

	final Composite transparency050 = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.50f );
	
	/*
	 *  the files must be in the resources/feature folder
	 */
	private static final Icon uploadIcon = new ImageIcon(FeaturePanelNew.class.getResource("upload.png"));
	private static final Icon downloadIcon = new ImageIcon(FeaturePanelNew.class.getResource("download.png"));
 

	/** This {@link ActionEvent} is fired when the 'next' button is pressed. */
	private ActionEvent NEXT_BUTTON_PRESSED = new ActionEvent( this, 0, "Next" );
	/** This {@link ActionEvent} is fired when the 'previous' button is pressed. */
	private ActionEvent PREVIOUS_BUTTON_PRESSED = new ActionEvent( this, 1, "Previous" );
	private ActionEvent ADDCLASS_BUTTON_PRESSED = new ActionEvent( this, 2, "AddClass" );
	private ActionEvent SAVECLASS_BUTTON_PRESSED= new ActionEvent( this, 3, "SaveLabel" );
	private ActionEvent DELETE_BUTTON_PRESSED = new ActionEvent( this, 4, "DeleteClass" );
	private ActionEvent COMPUTE_BUTTON_PRESSED  = new ActionEvent( this, 5, "TRAIN" );
	private ActionEvent SAVE_BUTTON_PRESSED  = new ActionEvent( this, 6, "SAVEDATA" );
	private ActionEvent TOGGLE_BUTTON_PRESSED = new ActionEvent( this, 7, "TOGGLE" );
	private ActionEvent DOWNLOAD_BUTTON_PRESSED = new ActionEvent( this, 8, "DOWNLOAD" );
	private ActionEvent MASKS_BUTTON_PRESSED = new ActionEvent( this, 8, "MASKS" );
	private ActionEvent LABELS_UPLOAD = new ActionEvent( this, 9, "LABELS" );
 

	private ImagePlus displayImage;
	/** Used only in classification setting, in segmentation we get from feature manager*/
	//private ImagePlus tempClassifiedImage;
	private JPanel imagePanel,classPanel,roiPanel;
	private JTextField imageNum;
	private JLabel total;
	private List<JCheckBox> jCheckBoxList;
	private Map<String,JTextArea> jTextList;
	private JComboBox<LearningType> learningType;
	private JFrame frame;
	private Button labels;
	ProjectManager projectManager = new ProjectManager();
	String labelPath = projectManager.getMetaInfo().getProjectDirectory().get(ASCommon.DEEPLEARNINGDIR) + "/label";

	/*
	 * constructor 
	 */
	public FeaturePanelNew(FeatureManager featureManager) {		
		super(featureManager.getCurrentImage());
		this.featureManager = featureManager;
		this.displayImage= featureManager.getCurrentImage();
		this.jCheckBoxList= new ArrayList<JCheckBox>();
		this.jTextList= new HashMap<String,JTextArea>();
		this.exampleList = new HashMap<String, JList<String>>();
		this.allexampleList = new HashMap<String, JList<String>>();
		roiOverlayList = new HashMap<String, RoiListOverlay>();
		//tempClassifiedImage = new ImagePlus();		
		this.setVisible(false);
		showPanel();
	}


	public void showPanel() {
		frame = new JFrame("Marking");	     
		
		frame.setResizable(false);
 		frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
		
		JList<String> frameList= GuiUtil.model();
		frameList.setForeground(Color.BLACK);
		
		
		JPanel panel = new JPanel();
		panel.setLayout(null);
		panel.setFont(panelFONT);
		panel.setBackground(Color.GRAY);
		
		imagePanel = new JPanel();	
		roiPanel= new JPanel();
		classPanel= new JPanel();
//		labels = new Button("Upload labels");

		/*
		 * image panel
		 */
		imagePanel.setLayout(new BorderLayout());
		
		ic=new SimpleCanvas(featureManager.getCurrentImage());
		ic.setMinimumSize(new Dimension(IMAGE_CANVAS_DIMENSION, IMAGE_CANVAS_DIMENSION));
		loadImage(displayImage);
		setOverlay();
		imagePanel.setBackground(Color.GRAY);
		imagePanel.add(ic,BorderLayout.CENTER);
		imagePanel.setBounds( 10, 10, IMAGE_CANVAS_DIMENSION, IMAGE_CANVAS_DIMENSION );		
		panel.add(imagePanel);
		
		/*
		 * class panel
		 */
	 	
		classPanel.setBounds(605,20,350,100);
		classPanel.setPreferredSize(new Dimension(350, 100));
		classPanel.setBorder(BorderFactory.createTitledBorder("Classes"));
		
		JScrollPane classScrolPanel = new JScrollPane(classPanel);
		classScrolPanel.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
		classScrolPanel.setBounds(605,20,350,80);
		addClassPanel();
		panel.add(classScrolPanel);
		
		
		/*
		 * features
		 */
		JPanel features= new JPanel();
		features.setBounds(605,120,350,100);
		features.setBorder(BorderFactory.createTitledBorder("Learning"));
		
		addButton(new JButton(), "PREVIOUS",null , 610, 130, 120, 20,features,PREVIOUS_BUTTON_PRESSED,null );
		
		imageNum= new JTextField();
		imageNum.setColumns(5);
		imageNum.setBounds( 630, 130, 10, 20 );
		JLabel dasedLine= new JLabel("/");
		dasedLine.setFont(new Font( "Arial", Font.PLAIN, 15 ));
		dasedLine.setForeground(Color.BLACK);
		dasedLine.setBounds(  670, 130, 10, 20 );
		total= new JLabel("Total");
		total.setFont(new Font( "Arial", Font.PLAIN, 15 ));
		total.setForeground(Color.BLACK);
		total.setBounds( 500, 600, 80, 30);		
		imageNum.setText(Integer.toString(featureManager.getCurrentSlice()));
		total.setText(Integer.toString(featureManager.getTotalSlice()));
		features.add(imageNum);
		features.add(dasedLine);
		features.add(total);
		
		/*
		 * compute panel
		 */
		
		JPanel computePanel = new JPanel();
		addButton(new JButton(), "Next",null , 800, 130, 80, 20,features,NEXT_BUTTON_PRESSED,null );
		addButton(new JButton(), "Train",null, 550,550,350,100,computePanel, COMPUTE_BUTTON_PRESSED,null);
		addButton(new JButton(), "Save",null, 550,550,350,100,computePanel, SAVE_BUTTON_PRESSED,null);
		addButton(new JButton(), "Overlay",null, 550,550,350,100,computePanel, TOGGLE_BUTTON_PRESSED,null);
		addButton(new JButton(), "Masks",null, 550,550,350,100,computePanel, MASKS_BUTTON_PRESSED,null);
		addButton(new JButton(), "Labels", null, 800,240,100,50, computePanel, LABELS_UPLOAD, null);
		
		features.add(computePanel);
		frame.add(features);
		
		/*
		 *  Data panel
		 */
		
		JPanel dataJPanel = new JPanel();
		learningType = new JComboBox<LearningType>(LearningType.values());
		learningType.setVisible(true);
		learningType.addItemListener( new ItemListener() {

			@Override
			public void itemStateChanged(ItemEvent e) {
				if(featureManager.getProjectType()==ProjectType.CLASSIF) {
					if(showColorOverlay) {
						updateGui();
						updateResultOverlay(null);
					} else 
						updateGui();			
				} else 
					updateGui();


				// here we need to add for classification
			}
		});
		
		dataJPanel.setBounds(700,240,100,60);
		learningType.setSelectedIndex(0);
		learningType.setFont( panelFONT );
		learningType.setBackground(Color.GRAY);
		learningType.setForeground(Color.WHITE);
		dataJPanel.add(learningType);
		dataJPanel.setBackground(Color.GRAY);
		
		panel.add(dataJPanel);

		/*
		 * labels
		 */

//		labels.setBounds(800,240,100,50);
//		labels.setForeground(Color.white);
//		learningType.setFont( panelFONT );
//		learningType.setBackground(Color.GRAY);
//		learningType.setForeground(Color.WHITE);
//
//		panel.add(labels);

		
		/*
		 * ROI panel
		 */
		roiPanel.setBorder(BorderFactory.createTitledBorder("Regions Of Interest"));
		//roiPanel.setPreferredSize(new Dimension(350, 400));
		JScrollPane scrollPane = new JScrollPane(roiPanel);
		scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);	
		scrollPane.setBounds(605,300,350,250);
		panel.add(scrollPane);
		frame.add(panel);
		
		/*
		 *  frame code
		 */
		frame.pack();
		frame.setSize(largeframeWidth,largeframeHight);
		//frame.setSize(getMaximumSize());		
		frame.setLocationRelativeTo(null);
		frame.setVisible(true);
		updateGui();

	}

	private void addClassPanel(){
		classPanel.removeAll();
		roiPanel.removeAll();
		jCheckBoxList.clear();
		jTextList.clear();
		int classes=featureManager.getNumOfClasses();
		IJ.log(Integer.toString(classes));
		if(classes%3==0){
			int tempSize=classes/3;
			classPanel.setPreferredSize(new Dimension(340, 80+30*tempSize));	
		}
		roiPanel.setPreferredSize(new Dimension(350, 175*classes));
		addButton(new JButton(), "ADD CLASS",null , 630, 20, 130, 20,classPanel,ADDCLASS_BUTTON_PRESSED,null );
		addButton(new JButton(), "SAVE CLASS",null , 630, 20, 130, 20,classPanel,SAVECLASS_BUTTON_PRESSED,null );
		addButton(new JButton(), "DELETE CLASS",null , 630, 20, 130, 20,classPanel,DELETE_BUTTON_PRESSED,null );
		for(String key: featureManager.getClassKeys()){
			String label=featureManager.getClassLabel(key);
			Color color= featureManager.getClassColor(key);
			addClasses(key,label,color);
			addSidePanel(color,key,label);
		}		
	}

	/**
	 * Draw the painted traces on the display image
	 */
	private void drawExamples(){
		for(String key: featureManager.getClassKeys()){
			ArrayList<Roi> rois=(ArrayList<Roi>) featureManager.
					getExamples(key,learningType.getSelectedItem().toString(), featureManager.getCurrentSlice());
			roiOverlayList.get(key).setColor(featureManager.getClassColor(key));
			roiOverlayList.get(key).setRoi(rois);
			//System.out.println("roi draw"+ key);
		}

		getImagePlus().updateAndDraw();
	}
	private void addSidePanel(Color color,String key,String label){
		JPanel panel= new JPanel();
		JList<String> current=GuiUtil.model();

		current.setForeground(color);
		exampleList.put(key,current);
		JList<String> all=GuiUtil.model();
		all.setForeground(color);
		allexampleList.put(key,all);	
		RoiListOverlay roiOverlay = new RoiListOverlay();
		roiOverlay.setComposite( transparency050 );
		((OverlayedImageCanvas)ic).addOverlay(roiOverlay);
		roiOverlayList.put(key,roiOverlay);
		JPanel buttonPanel= new JPanel();
		buttonPanel.setName(key);
		ActionEvent addbuttonAction= new ActionEvent(buttonPanel, 1,"AddButton");
		ActionEvent uploadAction= new ActionEvent(buttonPanel, 2,"UploadButton");
		ActionEvent downloadAction= new ActionEvent(buttonPanel, 3,"DownloadButton");
		JButton addButton= new JButton();
		addButton.setName(key);
		JButton upload= new JButton();
		upload.setName(key);
		JButton download= new JButton();
		download.setName(key);
		addButton(addButton, label, null, 605,280,350,250, buttonPanel, addbuttonAction, null);
		addButton(upload, null, uploadIcon, 605,280,350,250, buttonPanel, uploadAction, null);
		addButton(download, null, downloadIcon, 605,280,350,250, buttonPanel, downloadAction, null);
		roiPanel.add(buttonPanel);
		panel.add(GuiUtil.addScrollPanel(exampleList.get(key),null));
		panel.add(GuiUtil.addScrollPanel(allexampleList.get(key),null));
		roiPanel.add(panel );
		exampleList.get(key).addMouseListener(mouseListener);
		allexampleList.get(key).addMouseListener(mouseListener);
	}

	private void addClasses(String key , String label, Color color){
		JCheckBox  checkBox = new JCheckBox();
		checkBox.setName(key);
		jCheckBoxList.add(checkBox);
		JTextArea textArea= new JTextArea();
		textArea.setName(key);
		textArea.setText(label );
		jTextList.put(key, textArea);
		classPanel.add(checkBox);
		classPanel.add(textArea);
		JButton button= new JButton();
		button.setBackground(color);
		button.setName(key);
		ActionEvent colorAction= new ActionEvent(button, color.getRGB(),"ColorButton");
		addAction(button, colorAction);		
		classPanel.add(button);
	}

	private void addAction(JButton button ,final  ActionEvent action){
		 button.addActionListener( new ActionListener()	{
			@Override
			public void actionPerformed( final ActionEvent e )	{
				try {
					doAction(action);
				} catch (IOException ioException) {
					ioException.printStackTrace();
				}
			}
		} );
	 
	}
	
	private void loadImage(ImagePlus image){
		this.displayImage=image;
		setImage(this.displayImage);
		updateImage(this.displayImage);
	}

	public void validateFrame(){
		frame.invalidate();
		frame.revalidate();
		frame.repaint();
	}

	public void doAction( final ActionEvent event ) throws IOException {
		if(event== ADDCLASS_BUTTON_PRESSED){
			featureManager.addClass();
			addClassPanel();
			validateFrame();
			updateGui();
		} // end if
		if(event==DELETE_BUTTON_PRESSED){          

			System.out.println(featureManager.getNumOfClasses());
			System.out.println(jCheckBoxList.size());
			int totalDel=0;
			
			for (JCheckBox checkBox : jCheckBoxList) 
				if (checkBox.isSelected()) 
					totalDel++;
		
			if(featureManager.getNumOfClasses()-totalDel<2) 
             JOptionPane.showMessageDialog(null, "There should be minimum two classes");
			else {
				for (JCheckBox checkBox : jCheckBoxList) 
					if (checkBox.isSelected()) 
						featureManager.deleteClass(checkBox.getName());
				addClassPanel();
				validateFrame();
				updateGui();
			}	

		} // end if

		if(event==SAVE_BUTTON_PRESSED){
			featureManager.saveFeatureMetadata();
			JOptionPane.showMessageDialog(null, "Successfully saved regions of interest");

		} //end if
		
		if(event==SAVECLASS_BUTTON_PRESSED){
			for (JCheckBox checkBox : jCheckBoxList) {				
				//System.out.println(checkBox.getText());
				String key=checkBox.getName();
				featureManager.setClassLabel(key,jTextList.get(key).getText() );
				
			}
			addClassPanel();
			validateFrame();
			updateGui();
		} // end if
		
		if(event == PREVIOUS_BUTTON_PRESSED){			
			ImagePlus image=featureManager.getPreviousImage();
			imageNum.setText(Integer.toString(featureManager.getCurrentSlice()));
			loadImage(image);
			updateResultOverlay(image);
			
			if (showColorOverlay){
				if(featureManager.getProjectType()==ProjectType.CLASSIF) 
					classifiedImage = null;
				else 
					classifiedImage=featureManager.getClassifiedImage();		
				updateResultOverlay(classifiedImage);
			}

			// force limit size of image window
			if(ic.getWidth()>IMAGE_CANVAS_DIMENSION) {
				int x_centre = ic.getWidth()/2+ic.getX();
				int y_centre = ic.getHeight()/2+ic.getY();
				ic.zoomIn(x_centre,y_centre);
			}			
			updateGui();
		} // end if
		
		if(event==NEXT_BUTTON_PRESSED  ){			
			ImagePlus image=featureManager.getNextImage();
			imageNum.setText(Integer.toString(featureManager.getCurrentSlice()));
			loadImage(image);
			updateResultOverlay(image);
			if (showColorOverlay){
				if(featureManager.getProjectType()==ProjectType.CLASSIF)
					classifiedImage = null;
				else
					classifiedImage=featureManager.getClassifiedImage();
				updateResultOverlay(classifiedImage);
			}

			// force limit size of image window
			if(ic.getWidth()>IMAGE_CANVAS_DIMENSION) {
				int x_centre = ic.getWidth()/2+ic.getX();
				int y_centre = ic.getHeight()/2+ic.getY();
				ic.zoomIn(x_centre,y_centre);
			}
			//imagePanel.add(ic);
			updateGui();
		} // end if
		
		if(event==COMPUTE_BUTTON_PRESSED){
			if(featureManager.getProjectType()==ProjectType.CLASSIF) {
				// it means new round of training, so set result setting to false
				showColorOverlay = false;
				// removing previous markings and reset things
				predictionResultClassification = null;
				displayImage.setOverlay(null);

				// compute new predictions
				featureManager.compute();				
				predictionResultClassification = featureManager.getClassificationResultMap();

				// we do not need to get any image in classification setting, only predictions are needed
				classifiedImage = null;
			}

			//segmentation setting
			else {
				classifiedImage=featureManager.compute();
			}
			IJ.log("compute");

			toggleOverlay();
		} //end if
		
		if(event==TOGGLE_BUTTON_PRESSED){
			toggleOverlay();
		} // end if
		
		if(event==DOWNLOAD_BUTTON_PRESSED){

			ImagePlus image=featureManager.stackedClassifiedImage();
			image.show();
			//FileSaver saver= new FileSaver(image);
			//saver.saveAsTiff();
		} //end if
		
		if(event==MASKS_BUTTON_PRESSED){
			System.out.println("masks ");
			if (classifiedImage==null) {
				classifiedImage=featureManager.compute();
			}
			classifiedImage.show();
			 
		} //end if

		if(event==LABELS_UPLOAD){
			JFileChooser chooser = new JFileChooser();
			chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
			if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
				File f = chooser.getSelectedFile();
				ProjectManager projectManager = new ProjectManager();
				ProjectInfo projectInfo = projectManager.getMetaInfo();
				String path = projectInfo.getProjectDirectory().get(ASCommon.LABELSDIR);
//				FileUtils.copyDirectory(f, dir);
				for (File file: f.listFiles()){
					FileUtils.copyFile(file, new File(path+"/"+ FilenameUtils.getBaseName(file.getName())+"1.png"));
				}
				updateResultOverlay(featureManager.getCurrentImage());

			}
			selectFile(0.8);

			System.out.println("Labels uploaded");
		} //end if
		
		if(event.getActionCommand()== "ColorButton"){	
			String key=((Component)event.getSource()).getName();
			Color c;
			c = JColorChooser.showDialog( new JFrame(),
					"CLASS COLOR", featureManager.getClassColor(key));

			((Component)event.getSource()).setBackground(c);
			featureManager.updateColor(key, c);
			updateGui();
		}// end if
		
		if(event.getActionCommand()== "AddButton"){	
			String key=((Component)event.getSource()).getName();
			final Roi r = displayImage.getRoi();
			if (null == r)
				return;
			displayImage.killRoi();
			
			if(featureManager.addExample(key,r,learningType.getSelectedItem().toString(),featureManager.getCurrentSlice()))
				updateGui();
			else 
			    JOptionPane.showMessageDialog(null, "Other class already contain roi");	
	
			
		} //end if
		
		if(event.getActionCommand()== "UploadButton"){	
			String key=((Component)event.getSource()).getName();
			uploadExamples(key);
			updateGui();
		}//end if
		
		if(event.getActionCommand()== "DownloadButton"){	
			String key=((Component)event.getSource()).getName();
			downloadRois(key);
		}


	}


	/**
	 * Toggle between overlay and original image with markings
	 */
	private void toggleOverlay()
	{
		if(featureManager.getProjectType()== ProjectType.SEGM) {
			showColorOverlay = !showColorOverlay;			
			if (showColorOverlay && (null != classifiedImage)){
				updateResultOverlay(classifiedImage);
			}
			else{
				resultOverlay.setImage(null);
				displayImage.updateAndDraw();
			}
		}

		// classification setting, no classified image
		else {			
			showColorOverlay = !showColorOverlay;
			// user wants to see results
			if(showColorOverlay) {
				updateResultOverlay(classifiedImage);
			}

			// user wants to see original rois, no results
			else {

				// remove result overlay
				displayImage.setOverlay(null);
				displayImage.updateAndDraw();

				//just show examples drawn by user
				updateGui();
			}
		}		
	}

	public void updateResultOverlay(ImagePlus classifiedImage)
	{
		if(featureManager.getProjectType()==ProjectType.SEGM) {
			System.out.println(classifiedImage.getTitle());
			ProjectManager projectManager = new ProjectManager();
			ProjectInfo projectInfo = projectManager.getMetaInfo();
			BufferedImage fgImage = readImage(projectInfo.getProjectDirectory().get(ASCommon.LABELSDIR)+ "/" + classifiedImage.getTitle());
			ImagePlus imagePlus = new ImagePlus("label", fgImage);
			ImageProcessor overlay = imagePlus.getProcessor();
			overlay = overlay.convertToByte(false);
			setLut(featureManager.getColors());
			overlay.setColorModel(overlayLUT);
			resultOverlay.setImage(overlay);
//			ImagePlus imagePlus1 = new ImagePlus();
//			ImageProcessor imageProcessor = imagePlus1.getProcessor();
//			imageProcessor.setColorModel(overlayLUT);
//			imageProcessor.fill();
//			resultOverlay.setImage(imageProcessor);
			displayImage.updateAndDraw();

		}

		if(featureManager.getProjectType()== ProjectType.CLASSIF) {
			// remove previous overlay
			displayImage.setOverlay(null);
			displayImage.updateAndDraw();

			//get current slice
			int currentSlice = featureManager.getCurrentSlice();			
			Font font = new Font("Arial", Font.PLAIN, 38);           
			Overlay overlay = new Overlay();		 		 			 			
			ArrayList<Roi> rois;
			for(String classKey:featureManager.getClassKeys()) {
				//returns rois of current image slice of given class, current slice is updated internally
				rois = (ArrayList<Roi>) featureManager.getExamples(classKey,learningType.getSelectedItem().toString(), featureManager.getCurrentSlice());
				if(rois!=null) {					
					for (Roi roi:rois) {
						int pred = predictionResultClassification.get(roi.getName());
						TextRoi textroi = new TextRoi(roi.getBounds().x,roi.getBounds().y,
								roi.getFloatWidth(),roi.getFloatHeight(),Integer.toString(pred),font);
						textroi.setFillColor(roi.getFillColor());
						//textroi.setNonScalable(true);
						textroi.setPosition(currentSlice);
						overlay.add(textroi);
					}
				}
			}
			// add result overlay
			displayImage.setOverlay(overlay);			
			displayImage.updateAndDraw();				
		}
	}

	public void setLut(List<Color> colors ){
		int i=0;
		for(Color color: colors){
			red[i] = (byte) color.getRed();
			green[i] = (byte) color.getGreen();
			blue[i] = (byte) color.getBlue();
			i++;
		}
		overlayLUT = new LUT(red, green, blue);
	}
	
	private void updateGui(){
		try{
			drawExamples();
			updateExampleLists();
			//updateallExampleLists();
			ic.setMinimumSize(new Dimension(IMAGE_CANVAS_DIMENSION, IMAGE_CANVAS_DIMENSION));
			ic.repaint();
		}catch(Exception e){
			e.printStackTrace();
		}
	}

	private void updateExampleLists()	{
		LearningType type=(LearningType) learningType.getSelectedItem();
		for(String key:featureManager.getClassKeys()){
			exampleList.get(key).removeAll();
			Vector<String> listModel = new Vector<String>();

			for(int j=0; j<featureManager.getRoiListSize(key, learningType.getSelectedItem().toString(),featureManager.getCurrentSlice()); j++){	
				listModel.addElement(key+ " "+ j + " " +
						featureManager.getCurrentSlice()+" "+type.getLearningType());
			}
			exampleList.get(key).setListData(listModel);
			exampleList.get(key).setForeground(featureManager.getClassColor(key));
		}
	}	

	private  MouseListener mouseListener = new MouseAdapter() {
		public void mouseClicked(MouseEvent mouseEvent) {
			JList<?>  theList = ( JList<?>) mouseEvent.getSource();
			if (mouseEvent.getClickCount() == 1) {
				int index = theList.getSelectedIndex();

				if (index >= 0) {
					String item =theList.getSelectedValue().toString();
					String[] arr= item.split(" ");
					//System.out.println("Class Id"+ arr[0].trim());
					//int sliceNum=Integer.parseInt(arr[2].trim());
					showSelected( arr[0].trim(),index);

				}
			}

			if (mouseEvent.getClickCount() == 2) {
				int index = theList.getSelectedIndex();
				String type= learningType.getSelectedItem().toString();
				if (index >= 0) {
					String item =theList.getSelectedValue().toString();
					//System.out.println("ITEM : "+ item);
					String[] arr= item.split(" ");
					//int classId= featureManager.getclassKey(arr[0].trim())-1;
					featureManager.deleteExample(arr[0], Integer.parseInt(arr[1].trim()), type);
					updateGui();
				}
			}
		}
	};


	/**
	 * Select a list and deselect the others
	 * @param e item event (originated by a list)
	 * @param i list index
	 */
	private void showSelected(String classKey,int index ){
		updateGui();


		displayImage.setColor(Color.YELLOW);
		String type= learningType.getSelectedItem().toString();
		//System.out.println(classKey+"--"+index+"---"+type);
		final Roi newRoi = featureManager.getRoi(classKey, index,type);	
		//System.out.println(newRoi);
		newRoi.setImage(displayImage);
		displayImage.setRoi(newRoi);
		displayImage.updateAndDraw();
	}  
	private JButton addButton(final JButton button ,final String label, final Icon icon, final int x,
			final int y, final int width, final int height,
			JComponent panel, final ActionEvent action,final Color color )
	{
		panel.add(button);
		button.setText( label );
		button.setIcon( icon );
		button.setFont( panelFONT );
		button.setBorderPainted(false); 
		button.setFocusPainted(false); 
		button.setBackground(new Color(192, 192, 192));
		button.setForeground(Color.WHITE);
		if(color!=null){
			button.setBackground(color);
		}
		button.setBounds( x, y, width, height );
		button.addActionListener( new ActionListener()
		{
			@Override
			public void actionPerformed( final ActionEvent e )
			{
				//System.out.println(e.toString());
				try {
					doAction(action);
				} catch (IOException ioException) {
					ioException.printStackTrace();
				}
			}
		});

		return button;
	}

	private void setOverlay(){
		resultOverlay = new ImageOverlay();
		resultOverlay.setComposite( overlayAlpha );
		((OverlayedImageCanvas)ic).addOverlay(resultOverlay);
	}

	private void downloadRois(String key) {
		String type=learningType.getSelectedItem().toString();
		JFileChooser fileChooser = new JFileChooser();
		fileChooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
		fileChooser.setAcceptAllFileFilterUsed(false);
		int rVal = fileChooser.showOpenDialog(null);
		if (rVal == JFileChooser.APPROVE_OPTION) {
			String name=fileChooser.getSelectedFile().toString();
			if(!name.endsWith(".zip")){
				name = name + ".zip";
			}

			featureManager.saveExamples(name, key,type, featureManager.getCurrentSlice());
		}
	}

	private void uploadExamples(String key) {
		String type=learningType.getSelectedItem().toString();
		JFileChooser fileChooser = new JFileChooser();
		fileChooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
		fileChooser.setAcceptAllFileFilterUsed(false);
		int rVal = fileChooser.showOpenDialog(null);
		if (rVal == JFileChooser.APPROVE_OPTION) {
			featureManager.uploadExamples(fileChooser.getSelectedFile().toString(),key,type, featureManager.getCurrentSlice());
		}
	}

	public void selectFile(double percentage) throws IOException {

			ProjectManager projectManager = new ProjectManager();
			ProjectInfo projectInfo = projectManager.getMetaInfo();
			File f = new File(projectInfo.getProjectDirectory().get(ASCommon.LABELSDIR));
			File m = new File(projectInfo.getProjectDirectory().get(ASCommon.DEEPLEARNINGDIR));
			new File(m+"/label").mkdirs();
			File images = new File (m+"/image");
			File labels = new File(m+"/label");
			FileUtils.copyDirectory(f, labels);
			FileUtils.copyDirectory(new File(projectInfo.getProjectDirectory().get(ASCommon.IMAGESDIR)), images);
			System.out.println(projectInfo.getProjectDirectory().get(ASCommon.IMAGESDIR));
			File[] imagesArr = images.listFiles();
			File[] labelsArr = labels.listFiles();
			Arrays.sort(imagesArr);
			Arrays.sort(labelsArr);
			for (int i = 0; i < imagesArr.length; i++) {
				imagesArr[i].renameTo(new File(images + "/" + i + ".png"));
			}
			for (int l = 0; l < labelsArr.length; l++){
				labelsArr[l].renameTo(new File(labels +"/"+ l + ".png"));
			}

			new File(m+"/train").mkdirs();
			new File(m+"/test").mkdirs();
			new File(m+"/train/image").mkdirs();
			new File(m+"/test/image").mkdirs();
			new File(m+"/train/label").mkdirs();
			new File(m+"/test/label").mkdirs();

			int index = (int) (imagesArr.length*percentage);
			for (int j = 0; j < index; j++){
				FileUtils.copyFileToDirectory(new File(images + "/" + j + ".png"), new File(m+"/train/image"));

			}
			for (int j = 0; j < index; j++){
				FileUtils.copyFileToDirectory(new File(labels + "/" + j+ ".png"), new File(m+"/train/label"));

			}
			for (int g = index; g < imagesArr.length ; g++){
				FileUtils.copyFileToDirectory(new File(images + "/" + g+ ".png"), new File(m+"/test/image"));
			}
			for (int g = index; g < imagesArr.length ; g++){
				FileUtils.copyFileToDirectory(new File(labels + "/" + g+ ".png"), new File(m+"/test/label"));
			}
			FileUtils.deleteDirectory(images);
			FileUtils.deleteDirectory(labels);

	}

	public static BufferedImage readImage(String fileLocation) {
		BufferedImage img = null;
		try {
			img = ImageIO.read(new File(fileLocation));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return img;
	}

	public static ImagePlus invert(ImagePlus inputFile){
		for (int x = 0; x < inputFile.getWidth(); x++) {
			for (int y = 0; y < inputFile.getHeight(); y++) {
				int rgba = inputFile.getRGB(x, y);
				Color col = new Color(rgba, true);
				col = new Color(255 - col.getRed(), 255 - col.getGreen(),
						255 - col.getBlue());
				inputFile.setRGB(x, y, col.getRGB());
			}
		}
	}


}