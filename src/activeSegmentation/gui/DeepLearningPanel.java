package activeSegmentation.gui;

import activeSegmentation.ASCommon;
import activeSegmentation.IDeepLearning;
import activeSegmentation.deepLearning.UNetImplementation;
import activeSegmentation.feature.FeatureManager;
import activeSegmentation.learning.DeepLearningManager;
import activeSegmentation.prj.ProjectInfo;
import activeSegmentation.prj.ProjectManager;
import activeSegmentation.util.GuiUtil;
import org.apache.commons.io.FileUtils;
import weka.core.OptionHandler;
import weka.gui.GenericObjectEditor;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class DeepLearningPanel extends Component implements Runnable, ASCommon, ActionListener, PropertyChangeListener {
    private JList<String> modelList;
    private GenericObjectEditor m_ClassifierEditor = new GenericObjectEditor();
    private String originalOptions;
    String originalClassifierName;
    private ProjectManager projectManager;
    private ProjectInfo projectInfo;
    final JFrame frame = new JFrame("DEEP LEARNING");
    JList<String> featureSelList;
    final ActionEvent SAVE_BUTTON_PRESSED = new ActionEvent(this, 1, "Train");
    DeepLearningManager deepLearningManager;
    Button openButton;
    Button featureButton;
    JFileChooser fc;
    FeatureManager featureManager;
    int overlayOpacity = 33;

    Composite overlayAlpha = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, overlayOpacity / 100f);



    public DeepLearningPanel(ProjectManager projectManager, DeepLearningManager deepLearningManager, FeatureManager featureManager)  {
        this.projectManager = projectManager;
        this.deepLearningManager=deepLearningManager;
        this.featureManager = featureManager;
        this.projectInfo = projectManager.getMetaInfo();
        this.modelList = GuiUtil.model();
    }

    public void doAction(ActionEvent event) throws IOException {
        if (event == this.SAVE_BUTTON_PRESSED)     {
            UNetImplementation uNetImplementation = new UNetImplementation(projectInfo);
            uNetImplementation.importData();


            //System.out.println(this.featureSelList.getSelectedIndex());
//            this.projectInfo.setFeatureSelection((String)this.featureSelList.getSelectedValue());
//
//            // System.out.println("in set classifiler");
//            IDeepLearning testClassifier=setClassifier();
//
//            if(testClassifier!=null) {
//                IDeepLearning deepModel = new UNetImplementation();
//                this.deepLearningManager.setClassifier(deepModel);
//                this.projectManager.updateMetaInfo(this.projectInfo);
//            }

        }
    }

    public void actionPerformed(ActionEvent e) {

        //Handle open button action.
        if (e.getSource() == openButton) {
            int returnVal = fc.showOpenDialog(DeepLearningPanel.this);

            if (returnVal == JFileChooser.APPROVE_OPTION) {
                File file = fc.getSelectedFile();
                //This is where a real application would open the file.
                System.out.println(file.getName());
            } else {
                System.out.println("oops");
            }
            //Handle save button action.
        } else if (e.getSource() == featureButton) {
            int returnVal = fc.showSaveDialog(DeepLearningPanel.this);
            if (returnVal == JFileChooser.APPROVE_OPTION) {
                File file = fc.getSelectedFile();
                //This is where a real application would save the file.
                System.out.println(file.getName());
            } else {
                System.out.println("oops 2");
            }
        }
    }

    public void run()  {
        this.frame.setDefaultCloseOperation(1);
        this.frame.getContentPane().setBackground(Color.GRAY);
        this.frame.setLocationRelativeTo(null);
        this.frame.setSize(750, 400);
        JPanel learningP = new JPanel();
        learningP.setLayout(null);
        learningP.setBackground(Color.GRAY);

        JPanel learningJPanel = new JPanel();
        learningJPanel.setBorder(BorderFactory.createTitledBorder("Select the model"));
        String[] models = {"UNet", "OtherModel"};
        JList list = new JList(models);
        JScrollPane scrollPane = new JScrollPane(list);
        learningJPanel.add(scrollPane);
        learningJPanel.setBounds(30, 30, 250, 60);


        JPanel options = new JPanel();
        options.setBorder(BorderFactory.createTitledBorder("Learning Options"));
        options.setBounds(30, 120, 250, 80);


        Checkbox transferLearning = new Checkbox("Transfer learning");
        options.add(transferLearning);
        JPanel resetJPanel = new JPanel();
        resetJPanel.setBackground(Color.GRAY);
        resetJPanel.setBounds(370, 250, 200, 150);
        resetJPanel.add(addButton("TRAIN", null, 600, 500, 300, 50, this.SAVE_BUTTON_PRESSED));

        JPanel parametersPanel = new JPanel();
        parametersPanel.setBorder(BorderFactory.createTitledBorder("Learning Parameters"));
        parametersPanel.setBounds(370, 20, 300, 100);

        JLabel learningRateLabel = new JLabel("Learning rate:");
        JLabel numEpochsLabel = new JLabel("Number of epochs:");
        JLabel batchSizeLabel = new JLabel("Batch size:");


        JFormattedTextField learningRate = new JFormattedTextField();
        learningRate.setColumns(2);
        learningRate.addPropertyChangeListener("value", this);
        JFormattedTextField numEpochs = new JFormattedTextField();
        numEpochs.setColumns(2);
        numEpochs.addPropertyChangeListener("value", this);
        JFormattedTextField batchSize = new JFormattedTextField();
        batchSize.setColumns(2);
        batchSize.addPropertyChangeListener("value", this);




        learningRateLabel.setLabelFor(learningRateLabel);
        numEpochsLabel.setLabelFor(numEpochs);
        batchSizeLabel.setLabelFor(batchSize);

        parametersPanel.add(learningRateLabel);
        parametersPanel.add(learningRate);
        parametersPanel.add(numEpochsLabel);
        parametersPanel.add(numEpochs);
        parametersPanel.add(batchSizeLabel);
        parametersPanel.add(batchSize);


        JPanel importLabels = new JPanel();
        importLabels.setBorder(BorderFactory.createTitledBorder("Import labels"));
        JFileChooser fc = new JFileChooser();
        fc.setLayout(new BorderLayout());
        fc.setSize(800,500);
        JButton openButton = new JButton("Import labels");
        openButton.addActionListener(e -> {
            try {
                selectFile(0.8);
                ImageOverlay io = new ImageOverlay();
                io.setComposite( overlayAlpha );

            } catch (IOException ioException) {
                ioException.printStackTrace();
            }
        });
        JButton featureButton = new JButton("Create labels");
        featureButton.addActionListener(e ->{
            new FeaturePanelNew(featureManager);

            new File(projectInfo.getProjectDirectory().get(ASCommon.DEEPLEARNINGDIR) + "/labels").mkdirs();
            projectInfo.getProjectDirectory().get(ASCommon.FEATURESDIR);
        });

        fc.setVisible(true);
        fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        JPanel buttonPanel = new JPanel(); //use FlowLayout
        buttonPanel.add(openButton);
        buttonPanel.add(featureButton);

        openButton.setVisible(true);
        openButton.addActionListener(this);
        featureButton.addActionListener(this);
        File labels = new File("label");
        if(!labels.exists()){
            labels.mkdirs();
        }
        importLabels.add(buttonPanel);
//        importLabels.add(fc);
        importLabels.setBounds(370, 150, 300, 100);


        JPanel dataAugmentationPanel = new JPanel();
        dataAugmentationPanel.setBorder(BorderFactory.createTitledBorder("Data augmentation:"));
        dataAugmentationPanel.setBounds(30, 230, 250, 80);

        Checkbox flip = new Checkbox("Flip");
        Checkbox otherTransform = new Checkbox("Something else");
        dataAugmentationPanel.add(flip);
        dataAugmentationPanel.add(otherTransform);

        learningP.add(importLabels);
        learningP.add(parametersPanel);
        learningP.add(learningJPanel);
        learningP.add(resetJPanel);
        learningP.add(options);
        learningP.add(dataAugmentationPanel);


        this.frame.add(learningP);
        this.frame.setVisible(true);
    }
//    public void propertyChange(PropertyChangeEvent e) {
//        Object source = e.getSource();
//        if (source == amountField) {
//            amount = ((Number)amountField.getValue()).doubleValue();
//        } else if (source == rateField) {
//            rate = ((Number)rateField.getValue()).doubleValue();
//        } else if (source == numPeriodsField) {
//            numPeriods = ((Number)numPeriodsField.getValue()).intValue();
//        }
//
//        double payment = computePayment(amount, rate, numPeriods);
//        paymentField.setValue(new Double(payment));
//    }


    public void selectFile(double percentage) throws IOException {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            File f = chooser.getSelectedFile();
            File m = new File(projectInfo.getProjectDirectory().get(ASCommon.DEEPLEARNINGDIR));
            new File(m+"/label").mkdirs();
            File images = new File (m+"/image");
            File labels = new File(m+"/label");
            String imagesPath = images.getPath();
            String labelPath = labels.getPath();
            FileUtils.copyDirectory(f, labels);
            FileUtils.copyDirectory(new File(projectInfo.getProjectDirectory().get(ASCommon.IMAGESDIR)), images);
            System.out.println(projectInfo.getProjectDirectory().get(ASCommon.IMAGESDIR));
            File[] imagesArr = images.listFiles();
            File[] labelsArr = labels.listFiles();
            Arrays.sort(imagesArr);
            Arrays.sort(labelsArr);
            for (int i = 0; i < imagesArr.length; i++){
                imagesArr[i].renameTo(new File(images +"/" + i + ".png"));

            }

            for (int l = 0; l < labelsArr.length; l++){
                labelsArr[l].renameTo(new File(labels +"/"+ l + ".png"));
            }
            for (int c = 0; c < imagesArr.length; c++) {
                BufferedImage bgImage = readImage(imagesPath+"/"+c + ".png");
                BufferedImage fgImage = readImage(labelPath+"/"+c + ".png");

                overLay(bgImage,fgImage);

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
            new File(m+"/image").delete();
            new File(m+"/label").delete();

        } else {
            System.out.println("doesnt work");
        }
    }
    public BufferedImage overLay(BufferedImage bgImage, BufferedImage fgImage){

        Graphics2D g = bgImage.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON);

        float alpha = 0.33f;
        g.setComposite(AlphaComposite.SrcOver);
        g.drawImage(bgImage, 0, 0, null);
        g.setComposite(AlphaComposite.SrcOver.derive(alpha));
        g.drawImage(fgImage, 0, 0, null);

        g.dispose();
        JPanel gui = new JPanel(new GridLayout(1, 0, 5, 5));

        gui.add(new JLabel(new ImageIcon(bgImage)));
        gui.add(new JLabel(new ImageIcon(fgImage)));
        JOptionPane.showMessageDialog(null, gui);
        return bgImage;
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


    private IDeepLearning setClassifier()
    {
        Object c = this.m_ClassifierEditor.getValue();
        String options = "";
        String[] optionsArray = ((OptionHandler)c).getOptions();
        System.out.println(originalOptions);
//        if ((c instanceof OptionHandler)) {
//            options = Utils.joinOptions(optionsArray);
//        }
//        if ((!this.originalClassifierName.equals(c.getClass().getName())) ||
//                (!this.originalOptions.equals(options))) {
//            try
//            {
//                AbstractClassifier cls = (AbstractClassifier)c.getClass().newInstance();
//                cls.setOptions(optionsArray);
//                return cls;
//            }
//            catch (Exception ex)
//            {
//                ex.printStackTrace();
//            }
//        }
        return null;
    }

    private JButton addButton(String label, ImageIcon icon, int x, int y, int width, int height, ActionEvent SAVE_BUTTON_PRESSED)
    {
        JButton button = new JButton(label, icon);
        button.setFont(labelFONT);
        button.setBorderPainted(false);
        button.setFocusPainted(false);
        button.setBackground(new Color(192, 192, 192));
        button.setForeground(Color.WHITE);
        button.setBounds(x, y, width, height);
        button.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                try {
                    DeepLearningPanel.this.doAction(SAVE_BUTTON_PRESSED);
                } catch (IOException ioException) {
                    ioException.printStackTrace();
                }
            }
        });
        return button;
    }

    @Override
    public void propertyChange(PropertyChangeEvent propertyChangeEvent) {

    }
}