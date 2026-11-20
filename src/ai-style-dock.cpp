#include "ai-style-dock.hpp"
#include <obs-module.h>
#include <QApplication>
#include <QDesktopServices>
#include <QUrl>
#include <QHeaderView>
#include <QMessageBox>
#include <QFileInfo>
#include <QStandardPaths>
#include "core/latent-engine.hpp"
#include "core/proactive_agent.hpp"

#include <QVBoxLayout>
#include <QStyle>
#include <QPalette>
#include <QPainter>
#include <QFont>
#include <QFontDatabase>
#include "common/theme-helpers.hpp"

AIStyleDock *g_aiStyleDock = nullptr;

// Shared QSS uses live OBS palette roles; dynamic status colors are added below.
static constexpr char PALETTE_STYLE[] = R"(
QDockWidget { background: palette(window); border: none; }
QDockWidget::title { background: palette(button); color: palette(button-text); padding: 5px 8px; font-weight: 600; }
QWidget { background: palette(window); color: palette(window-text); }
QTabWidget::pane { background: palette(base); border: 1px solid palette(mid); }
QTabBar::tab { background: palette(button); color: palette(button-text); padding: 5px 9px; border: 1px solid palette(mid); margin-right: 2px; }
QTabBar::tab:selected { background: palette(highlight); color: palette(highlighted-text); }
QComboBox, QLineEdit, QTableWidget { background: palette(base); color: palette(text); border: 1px solid palette(mid); border-radius: 3px; padding: 4px 6px; }
QTableWidget { gridline-color: palette(mid); selection-background-color: palette(highlight); selection-color: palette(highlighted-text); }
QHeaderView::section { background: palette(button); color: palette(button-text); padding: 4px; border: none; border-right: 1px solid palette(mid); }
QLabel#previewLabel { background: palette(shadow); border: 1px solid palette(mid); border-radius: 3px; }
QLineEdit:focus, QComboBox:focus { border-color: palette(highlight); }
QSlider::groove:horizontal { background: palette(mid); height: 4px; border-radius: 2px; }
QSlider::handle:horizontal { background: palette(highlight); width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; }
QSlider::sub-page:horizontal { background: palette(highlight); border-radius: 2px; }
QPushButton { background: palette(button); color: palette(button-text); border: 1px solid palette(mid); border-radius: 3px; padding: 4px 8px; }
QPushButton:hover { border-color: palette(highlight); }
QPushButton:pressed, QPushButton#generateBtn { background: palette(highlight); color: palette(highlighted-text); }
QProgressBar { background: palette(button); border: none; border-radius: 2px; height: 3px; }
QProgressBar::chunk { background: palette(highlight); border-radius: 2px; }
)";

AIStyleDock::AIStyleDock(QWidget *parent)
    : QDockWidget("👁️ Sight Lab", parent)
{
    setObjectName("SightLabDock");
    setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
    setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    
    netManager = new QNetworkAccessManager(this);
    
    setupUI();
    applyThemeStyle();
    createAnimations();

    // Verify Proactive Agent compilation
    lll::agent::ProactiveAgent agent;
    (void)agent;
    
    // Stats refresh timer
    statsTimer = new QTimer(this);
    connect(statsTimer, &QTimer::timeout, this, &AIStyleDock::onRefreshStats);
    statsTimer->start(500);
    
    // Progress animation timer
    progressTimer = new QTimer(this);
    connect(progressTimer, &QTimer::timeout, this, &AIStyleDock::pulseProgress);
    
    g_aiStyleDock = this;
    
    // Initial Scan
    QTimer::singleShot(1000, this, &AIStyleDock::scanForSources);
}

AIStyleDock::~AIStyleDock() {
    g_aiStyleDock = nullptr;
}

void AIStyleDock::setupUI() {
    QWidget *wrapper = new QWidget(this);
    QVBoxLayout *wrapperLayout = new QVBoxLayout(wrapper);
    wrapperLayout->setContentsMargins(0, 0, 0, 0);
    wrapperLayout->setSpacing(0);
    
    // Tabs
    mainTabs = new QTabWidget(wrapper);
    mainTabs->setDocumentMode(true);
    
    controlWidget = new QWidget();
    setupControlTab();
    mainTabs->addTab(controlWidget, "Control");
    
    managerWidget = new QWidget();
    setupManagerTab();
    mainTabs->addTab(managerWidget, "Manager");
    
    wrapperLayout->addWidget(mainTabs);
    setWidget(wrapper);
    setMinimumWidth(240);
}

void AIStyleDock::setupControlTab() {
    QVBoxLayout *mainLayout = new QVBoxLayout(controlWidget);
    mainLayout->setContentsMargins(6, 6, 6, 6);
    mainLayout->setSpacing(4);

    const auto addSectionLabel = [this, mainLayout](const QString &text) {
        auto *label = new QLabel(text, controlWidget);
        label->setProperty("themeRole", "section");
        mainLayout->addWidget(label);
    };
    
    // ===== Source Selector =====
    addSectionLabel("SOURCE");
    QHBoxLayout *sourceRow = new QHBoxLayout();
    sourceSelector = new QComboBox(controlWidget);
    sourceSelector->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    connect(sourceSelector, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &AIStyleDock::onSourceSelected);
    
    refreshSourcesBtn = new QPushButton("Refresh", controlWidget);
    refreshSourcesBtn->setToolTip("Scan for running plugins");
    connect(refreshSourcesBtn, &QPushButton::clicked, this, &AIStyleDock::scanForSources);
    
    sourceRow->addWidget(sourceSelector);
    sourceRow->addWidget(refreshSourcesBtn);
    mainLayout->addLayout(sourceRow);
    
    // ===== Preview Area =====
    addSectionLabel("PREVIEW");
    previewLabel = new QLabel(controlWidget);
    previewLabel->setObjectName("previewLabel");
    previewLabel->setMinimumSize(160, 90);
    previewLabel->setMaximumHeight(135);
    previewLabel->setAlignment(Qt::AlignCenter);
    previewLabel->setText("No Signal");
    previewLabel->setScaledContents(true);
    mainLayout->addWidget(previewLabel);
    
    // ===== Status Row =====
    QHBoxLayout *statusRow = new QHBoxLayout();
    statusRow->setSpacing(8);
    
    fpsLabel = new QLabel("-- FPS", controlWidget);
    fpsLabel->setObjectName("fpsLabel");
    statusRow->addWidget(fpsLabel);
    
    vramLabel = new QLabel("VRAM: --", controlWidget);
    vramLabel->setObjectName("vramLabel");
    statusRow->addWidget(vramLabel);
    
    droppedLabel = new QLabel("Drop: 0", controlWidget);
    statusRow->addWidget(droppedLabel);
    
    statusRow->addStretch();
    mainLayout->addLayout(statusRow);
    
    // ===== Progress Bar =====
    progressBar = new QProgressBar(controlWidget);
    progressBar->setTextVisible(false);
    progressBar->setMinimum(0);
    progressBar->setMaximum(100);
    progressBar->setValue(0);
    progressBar->setFixedHeight(3);
    mainLayout->addWidget(progressBar);
    
    // ===== Status Label =====
    statusLabel = new QLabel("Ready", controlWidget);
    statusLabel->setObjectName("statusLabel");
    mainLayout->addWidget(statusLabel);
    
    // ===== Prompt Input =====
    addSectionLabel("CONTROLS");
    QLabel *promptLabel = new QLabel("Prompt", controlWidget);
    promptLabel->setProperty("themeRole", "muted");
    mainLayout->addWidget(promptLabel);
    
    promptEdit = new QLineEdit(controlWidget);
    promptEdit->setPlaceholderText("cinematic, dramatic lighting...");
    connect(promptEdit, &QLineEdit::editingFinished, this, &AIStyleDock::onPromptEdited);
    mainLayout->addWidget(promptEdit);
    
    // ===== Strength Slider =====
    QHBoxLayout *strengthRow = new QHBoxLayout();
    strengthRow->setSpacing(4);
    
    QLabel *strengthLabel = new QLabel("Str", controlWidget);
    strengthLabel->setProperty("themeRole", "muted");
    strengthRow->addWidget(strengthLabel);
    
    strengthSlider = new QSlider(Qt::Horizontal, controlWidget);
    strengthSlider->setRange(0, 100);
    strengthSlider->setValue(75);
    connect(strengthSlider, &QSlider::valueChanged, this, &AIStyleDock::onStrengthSliderChanged);
    strengthRow->addWidget(strengthSlider, 1);
    
    strengthValueLabel = new QLabel("0.75", controlWidget);
    strengthValueLabel->setFixedWidth(28);
    strengthValueLabel->setAlignment(Qt::AlignRight);
    strengthRow->addWidget(strengthValueLabel);
    
    mainLayout->addLayout(strengthRow);
    
    // ===== Steps Slider =====
    QHBoxLayout *stepsRow = new QHBoxLayout();
    stepsRow->setSpacing(4);
    
    QLabel *stepsLabel = new QLabel("Step", controlWidget);
    stepsLabel->setProperty("themeRole", "muted");
    stepsRow->addWidget(stepsLabel);
    
    stepsSlider = new QSlider(Qt::Horizontal, controlWidget);
    stepsSlider->setRange(1, 30);
    stepsSlider->setValue(8);
    connect(stepsSlider, &QSlider::valueChanged, this, &AIStyleDock::onStepsSliderChanged);
    stepsRow->addWidget(stepsSlider, 1);
    
    stepsValueLabel = new QLabel("8", controlWidget);
    stepsValueLabel->setFixedWidth(20);
    stepsValueLabel->setAlignment(Qt::AlignRight);
    stepsRow->addWidget(stepsValueLabel);
    
    mainLayout->addLayout(stepsRow);
    
    // ===== Preset Buttons =====
    QLabel *presetLabel = new QLabel("Quick Presets", controlWidget);
    presetLabel->setProperty("themeRole", "muted");
    mainLayout->addWidget(presetLabel);
    
    QHBoxLayout *presetRow = new QHBoxLayout();
    presetRow->setSpacing(3);
    
    animeBtn = new QPushButton("Anime", controlWidget);
    animeBtn->setToolTip("Anime");
    animeBtn->setProperty("preset", "Anime");
    connect(animeBtn, &QPushButton::clicked, this, &AIStyleDock::onPresetClicked);
    presetRow->addWidget(animeBtn);
    
    cineBtn = new QPushButton("Cinematic", controlWidget);
    cineBtn->setToolTip("Cinematic");
    cineBtn->setProperty("preset", "Cinematic");
    connect(cineBtn, &QPushButton::clicked, this, &AIStyleDock::onPresetClicked);
    presetRow->addWidget(cineBtn);
    
    cyberBtn = new QPushButton("Cyberpunk", controlWidget);
    cyberBtn->setToolTip("Cyberpunk");
    cyberBtn->setProperty("preset", "Cyberpunk");
    connect(cyberBtn, &QPushButton::clicked, this, &AIStyleDock::onPresetClicked);
    presetRow->addWidget(cyberBtn);
    
    zimageBtn = new QPushButton("Z-Image", controlWidget);
    zimageBtn->setToolTip("Z-Image Turbo");
    zimageBtn->setProperty("preset", "Z-Image Turbo");
    connect(zimageBtn, &QPushButton::clicked, this, &AIStyleDock::onPresetClicked);
    presetRow->addWidget(zimageBtn);
    
    mainLayout->addLayout(presetRow);
    
    // ===== Enhance Buttons =====
    QHBoxLayout *enhanceRow = new QHBoxLayout();
    enhanceRow->setSpacing(3);
    
    QPushButton *upscaleBtn = new QPushButton("Upscale", controlWidget);
    upscaleBtn->setToolTip("Real-ESRGAN x4");
    upscaleBtn->setProperty("preset", "Upscale");
    connect(upscaleBtn, &QPushButton::clicked, this, &AIStyleDock::onPresetClicked);
    enhanceRow->addWidget(upscaleBtn);
    
    QPushButton *fixLightBtn = new QPushButton("Fix light", controlWidget);
    fixLightBtn->setToolTip("Zero-DCE");
    fixLightBtn->setProperty("preset", "Fix Light");
    connect(fixLightBtn, &QPushButton::clicked, this, &AIStyleDock::onPresetClicked);
    enhanceRow->addWidget(fixLightBtn);
    
    mainLayout->addLayout(enhanceRow);
    
    // ===== Bottom Buttons =====
    QHBoxLayout *bottomRow = new QHBoxLayout();
    bottomRow->setSpacing(4);
    
    generateBtn = new QPushButton("Generate", controlWidget);
    generateBtn->setObjectName("generateBtn");
    connect(generateBtn, &QPushButton::clicked, this, &AIStyleDock::generateClicked);
    bottomRow->addWidget(generateBtn, 2);
    
    mainLayout->addLayout(bottomRow);
    mainLayout->addStretch();
}

void AIStyleDock::setupManagerTab() {
    QVBoxLayout *layout = new QVBoxLayout(managerWidget);
    layout->setContentsMargins(4, 4, 4, 4);

    auto *sectionLabel = new QLabel("MODEL MANAGER", managerWidget);
    sectionLabel->setProperty("themeRole", "section");
    layout->addWidget(sectionLabel);
    
    modelTable = new QTableWidget(managerWidget);
    modelTable->setColumnCount(3);
    modelTable->setHorizontalHeaderLabels({"Model", "Type", "Action"});
    modelTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    modelTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    modelTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Fixed);
    modelTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    modelTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    modelTable->setAlternatingRowColors(true);
    modelTable->verticalHeader()->setVisible(false);
    layout->addWidget(modelTable);
    
    downloadBar = new QProgressBar(managerWidget);
    downloadBar->setVisible(false);
    layout->addWidget(downloadBar);
    
    downloadStatusLabel = new QLabel(managerWidget);
    layout->addWidget(downloadStatusLabel);
    
    populateModelList();
}

void AIStyleDock::applyThemeStyle() {
    const auto colors = lll::ui::themeColors(this);
    QString style = QString::fromUtf8(PALETTE_STYLE);
    style += QString(R"(
 QLabel[themeRole="section"] { color: %1; font-size: 9px; font-weight: 700; letter-spacing: 1px; }
 QLabel[themeRole="muted"] { color: %2; font-size: 9px; }
 QLabel[themeRole="fps-good"] { color: %3; font-weight: 600; font-size: 11px; }
 QLabel[themeRole="fps-warn"] { color: %4; font-weight: 600; font-size: 11px; }
 QLabel[themeRole="fps-bad"] { color: %5; font-weight: 600; font-size: 11px; }
 QPushButton[themeRole="installed"] { background: %6; color: %7; border: none; }
 QPushButton[themeRole="active"] { background: %8; color: %7; }
     )").arg(colors.accent.name(), colors.mutedText.name(), colors.link.name(),
            colors.border.name(), colors.text.name(), colors.accent.name(),
            colors.accentText.name(), colors.accent.name());
    setStyleSheet(style);
}

void AIStyleDock::refreshTheme() {
    applyThemeStyle();
    update();
}

void AIStyleDock::createAnimations() {
    pulseAnim = new QPropertyAnimation(progressBar, "value", this);
    pulseAnim->setDuration(1000);
    pulseAnim->setLoopCount(-1);
}

// ============================================================================================
// LOGIC
// ============================================================================================

void AIStyleDock::populateSourceList() {
    // Save current selection ID to restore it
    QString currentName = sourceSelector->currentText();
    
    sourceSelector->clear();
    detectedSources.clear();
    
    // 1. Scan for our filters active in the system
    // We iterate over all sources and check for our filter IDs
    struct SourceCollector {
        std::vector<SourceItem> &sources;
        QString currentName;
        AIStyleDock *dock;
    };
    
    SourceCollector collector{detectedSources, currentName, this};
    
    auto enum_callback = [](void *param, obs_source_t *source) -> bool {
        SourceCollector *collector = static_cast<SourceCollector*>(param);
        
        const char *id = obs_source_get_id(source);
        const char *name = obs_source_get_name(source);
        
        // Is it a generator?
        bool isTarget = false;
        if (strcmp(id, "ai_image_generator") == 0) isTarget = true;
        
        // Is it a source with our filter?
        if (!isTarget) {
             obs_source_t *filter = obs_source_get_filter_by_name(source, "neural_enhance_filter");
             if (!filter) filter = obs_source_get_filter_by_name(source, "style_transfer_filter");
             
             if (filter) {
                 isTarget = true;
                 obs_source_release(filter);
             }
        }
        
        if (isTarget) {
            QString qname = QString::fromUtf8(name);
            collector->sources.push_back({qname, source});
            collector->dock->sourceSelector->addItem(qname);
            
            // If it was the previously selected one, re-select it
            if (qname == collector->currentName) {
                collector->dock->sourceSelector->setCurrentIndex(collector->dock->sourceSelector->count() - 1);
                collector->dock->connectedSource = source; // Re-establish pointer
            }
        }
        
        return true; // Continue enumeration
    };
    
    obs_enum_sources(enum_callback, &collector);
    
    if (sourceSelector->count() == 0) {
        sourceSelector->addItem("No Active AI Plugins");
        setStatus("No plugins found");
    } else {
        if (connectedSource == nullptr) {
             onSourceSelected(0);
        }
    }
}

void AIStyleDock::scanForSources() {
    populateSourceList();
    setStatus("Scanned for plugins");
}

void AIStyleDock::registerActiveSource(obs_source_t *source) {
    if (!source) return;
    // Just trigger a rescan to keep UI clean
    QMetaObject::invokeMethod(this, "scanForSources", Qt::QueuedConnection);
}

void AIStyleDock::unregisterActiveSource(obs_source_t *source) {
    if (source == connectedSource) {
        connectedSource = nullptr;
        previewLabel->setText("Source Disconnected");
    }
    QMetaObject::invokeMethod(this, "scanForSources", Qt::QueuedConnection);
}

void AIStyleDock::onSourceSelected(int index) {
    if (index < 0 || index >= (int)detectedSources.size()) return;
    
    SourceItem &item = detectedSources[index];
    
    // We need to verify the source pointer is still valid (OBS might have deleted it)
    // In a real plugin, we would hold weak refs or use names
    obs_source_t *source = obs_get_source_by_name(item.name.toUtf8().constData());
    if (source) {
        connectToSource(source);
        obs_source_release(source);
    } else {
        // Source gone
        populateSourceList();
    }
}

// ============================================================================================
// MODEL MANAGER LOGIC
// ============================================================================================

void AIStyleDock::populateModelList() {
    availableModels.clear();
    
    // Populate from MODELS.md definitions
    availableModels.append({"Z-Image-Turbo (Q4_0)", "Checkpoint", "z-image-turbo-q4_0.gguf", "https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z-image-turbo-q4_0.gguf", "High Quality, Fast", "checkpoints/z-image-turbo-q4_0.gguf"});
    availableModels.append({"Z-Image-Turbo (Q3_K)", "Checkpoint", "z-image-turbo-q3_k.gguf", "https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z-image-turbo-q3_k.gguf", "Lower VRAM (4GB)", "checkpoints/z-image-turbo-q3_k.gguf"});
    availableModels.append({"SDXL-Turbo (FP16)", "Checkpoint", "sdxl-turbo-fp16.gguf", "https://huggingface.co/leejet/SDXL-turbo-gguf/resolve/main/sdxl-turbo-fp16.gguf", "Fastest", "checkpoints/sdxl-turbo-fp16.gguf"});
    
    availableModels.append({"Canny ControlNet", "ControlNet", "control_v11p_sd15_canny.gguf", "https://huggingface.co/leejet/ControlNet-v1.1-GGUF/resolve/main/control_v11p_sd15_canny.gguf", "Edge Detection", "controlnet/control_v11p_sd15_canny.gguf"});
    availableModels.append({"Depth ControlNet", "ControlNet", "control_v11f1p_sd15_depth.gguf", "https://huggingface.co/leejet/ControlNet-v1.1-GGUF/resolve/main/control_v11f1p_sd15_depth.gguf", "Depth Map", "controlnet/control_v11f1p_sd15_depth.gguf"});
    
    availableModels.append({"TAESD (SD1.5)", "VAE", "taesd.safetensors", "https://huggingface.co/madebyollin/taesd/resolve/main/taesd.safetensors", "Speed Booster", "taesd/taesd.safetensors"});
    availableModels.append({"TAESD (SDXL)", "VAE", "taesdxl.safetensors", "https://huggingface.co/madebyollin/taesdxl/resolve/main/taesdxl.safetensors", "Speed Booster XL", "taesd/taesdxl.safetensors"});

    modelTable->setRowCount(availableModels.size());
    
    for(int i=0; i<availableModels.size(); ++i) {
        const auto& model = availableModels[i];
        
        QTableWidgetItem *nameItem = new QTableWidgetItem(model.name);
        nameItem->setToolTip(model.description);
        modelTable->setItem(i, 0, nameItem);
        
        QTableWidgetItem *typeItem = new QTableWidgetItem(model.type);
        modelTable->setItem(i, 1, typeItem);
        
        bool installed = isModelInstalled(model);
        
        QWidget *btnWidget = new QWidget();
        QHBoxLayout *btnLayout = new QHBoxLayout(btnWidget);
        btnLayout->setContentsMargins(2, 2, 2, 2);
        
        QPushButton *btn = new QPushButton(installed ? "Installed" : "Install");
        btn->setProperty("row", i);
        btn->setEnabled(!installed);
        if (installed) {
            btn->setProperty("themeRole", "installed");
        } else {
            btn->setObjectName("installBtn");
            connect(btn, &QPushButton::clicked, this, &AIStyleDock::onInstallModelClicked);
        }
        
        btnLayout->addWidget(btn);
        modelTable->setCellWidget(i, 2, btnWidget);
    }
}

QString AIStyleDock::getModelInstallPath(const QString &relativePath) {
    // We assume standard plugin path structure: <obs_plugin_data_path>/models/...
    // Since we don't have easy access to obs_module_get_config_path in this TU without globals
    // We will ask the C-side or assume standard location.
    
    // Fallback: Use standard OBS path construction if possible or a known path
    // On Windows: %APPDATA%/obs-studio/plugin_config/live-latent-lab-sight/models/
    
    QString configPath;
    char *c_path = obs_module_get_config_path(obs_current_module(), "");
    if (c_path) {
        configPath = QString::fromUtf8(c_path);
        bfree(c_path);
    } else {
        // Fallback for when module pointer isn't set (rare in dock context)
        // Try standard location
        configPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
        // This usually points to AppData/Roaming/<App>, need to adjust for OBS
    }
    
    // Assume relativePath starts with subdirectory (e.g. checkpoints/)
    return QDir(configPath).filePath("models/" + relativePath);
}

bool AIStyleDock::isModelInstalled(const ModelDef &model) {
    return QFile::exists(getModelInstallPath(model.localPath));
}

void AIStyleDock::onInstallModelClicked() {
    QPushButton *btn = qobject_cast<QPushButton*>(sender());
    if (!btn) return;
    
    int row = btn->property("row").toInt();
    if (row < 0 || row >= availableModels.size()) return;
    
    const ModelDef &model = availableModels[row];
    
    if (currentReply) {
        QMessageBox::warning(this, "Busy", "A download is already in progress.");
        return;
    }
    
    QString targetPath = getModelInstallPath(model.localPath);
    QFileInfo fileInfo(targetPath);
    QDir dir = fileInfo.absoluteDir();
    if (!dir.exists()) {
        dir.mkpath(".");
    }
    
    currentFile = new QFile(targetPath);
    if (!currentFile->open(QIODevice::WriteOnly)) {
        QMessageBox::critical(this, "Error", "Cannot write to: " + targetPath);
        delete currentFile;
        currentFile = nullptr;
        return;
    }
    
    QNetworkRequest request(QUrl(model.url));
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);
    
    currentReply = netManager->get(request);
    connect(currentReply, &QNetworkReply::downloadProgress, this, &AIStyleDock::onDownloadProgress);
    connect(currentReply, &QNetworkReply::finished, this, &AIStyleDock::onDownloadFinished);
    
    downloadBar->setVisible(true);
    downloadBar->setValue(0);
    downloadStatusLabel->setText("Downloading " + model.name + "...");
    
    btn->setText("...");
    btn->setEnabled(false);
}

void AIStyleDock::onDownloadProgress(qint64 bytesReceived, qint64 bytesTotal) {
    if (bytesTotal > 0) {
        downloadBar->setValue((int)((bytesReceived * 100) / bytesTotal));
        downloadStatusLabel->setText(QString("Downloading: %1%").arg((bytesReceived * 100) / bytesTotal));
    }
}

void AIStyleDock::onDownloadFinished() {
    if (currentFile) {
        currentFile->close();
        delete currentFile;
        currentFile = nullptr;
    }
    
    if (currentReply) {
        if (currentReply->error() != QNetworkReply::NoError) {
            QMessageBox::critical(this, "Download Failed", currentReply->errorString());
            downloadStatusLabel->setText("Error");
        } else {
            downloadStatusLabel->setText("Install Complete");
            // Refresh list to show "Installed"
            populateModelList();
        }
        currentReply->deleteLater();
        currentReply = nullptr;
    }
    
    downloadBar->setVisible(false);
}

// ============================================================================================
// EXISTING LOGIC (Preview, Stats, etc.)
// ============================================================================================

void AIStyleDock::updatePreview(const uint8_t* data, int width, int height) {
    if (!data || width <= 0 || height <= 0) return;
    
    std::lock_guard<std::mutex> lock(previewMutex);
    
    // Create QImage from RGBA data
    QImage img(data, width, height, width * 4, QImage::Format_RGBA8888);
    currentPreview = QPixmap::fromImage(img.scaled(
        previewLabel->size(),
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation
    ));
    
    QMetaObject::invokeMethod(previewLabel, [this]() {
        previewLabel->setPixmap(currentPreview);
    }, Qt::QueuedConnection);
}

void AIStyleDock::updateStats(float fps, int dropped, float vram_mb) {
    QMetaObject::invokeMethod(this, [this, fps, dropped, vram_mb]() {
        if (fps > 0) {
            fpsLabel->setText(QString("%1 FPS").arg(fps, 0, 'f', 1));
            // Color code FPS...
            fpsLabel->setProperty("themeRole", fps >= 3.0 ? "fps-good" :
                                                fps >= 1.5 ? "fps-warn" : "fps-bad");
            fpsLabel->style()->unpolish(fpsLabel);
            fpsLabel->style()->polish(fpsLabel);
        } else {
            fpsLabel->setText("-- FPS");
        }
        vramLabel->setText(QString("VRAM: %1G").arg(vram_mb / 1024.0, 0, 'f', 1));
        droppedLabel->setText(QString("Drop: %1").arg(dropped));
    }, Qt::QueuedConnection);
}

void AIStyleDock::updateProgress(float progress) {
    QMetaObject::invokeMethod(this, [this, progress]() {
        progressBar->setValue((int)(progress * 100));
        if (progress > 0 && progress < 1.0) {
            if (!progressTimer->isActive()) progressTimer->start(50);
        } else {
            progressTimer->stop();
        }
    }, Qt::QueuedConnection);
}

void AIStyleDock::setStatus(const QString &status) {
    QMetaObject::invokeMethod(this, [this, status]() {
        statusLabel->setText(status);
        QGraphicsOpacityEffect *effect = new QGraphicsOpacityEffect(statusLabel);
        statusLabel->setGraphicsEffect(effect);
        QPropertyAnimation *anim = new QPropertyAnimation(effect, "opacity", this);
        anim->setDuration(300);
        anim->setStartValue(0.5);
        anim->setEndValue(1.0);
        anim->start(QAbstractAnimation::DeleteWhenStopped);
    }, Qt::QueuedConnection);
}

void AIStyleDock::connectToSource(obs_source_t *source) {
    if (connectedSource) {
        disconnect(this, &AIStyleDock::presetSelected, this, &AIStyleDock::applyPresetToFilter);
    }
    
    connectedSource = source;
    if (!source) return;
    
    connect(this, &AIStyleDock::presetSelected, this, &AIStyleDock::applyPresetToFilter, Qt::UniqueConnection);
    
    // Setup signal connections for this source
    connect(this, &AIStyleDock::promptChanged, this, [this](const QString &prompt) {
        if (!connectedSource) return;
        obs_data_t *settings = obs_source_get_settings(connectedSource);
        obs_data_set_string(settings, "prompt", prompt.toUtf8().constData());
        obs_source_update(connectedSource, settings);
        obs_data_release(settings);
    });
    
    connect(this, &AIStyleDock::strengthChanged, this, [this](double strength) {
        if (!connectedSource) return;
        obs_data_t *settings = obs_source_get_settings(connectedSource);
        obs_data_set_double(settings, "strength", strength);
        obs_source_update(connectedSource, settings);
        obs_data_release(settings);
    });
    
    connect(this, &AIStyleDock::stepsChanged, this, [this](int steps) {
        if (!connectedSource) return;
        obs_data_t *settings = obs_source_get_settings(connectedSource);
        obs_data_set_int(settings, "steps", steps);
        obs_source_update(connectedSource, settings);
        obs_data_release(settings);
    });
    
    const char* name = obs_source_get_name(source);
    setStatus(QString("Connected: %1").arg(name));
    blog(LOG_INFO, "[AIStyleDock] Connected to: %s", name);
}

void AIStyleDock::onPromptEdited() { emit promptChanged(promptEdit->text()); }
void AIStyleDock::onStrengthSliderChanged(int value) {
    double strength = value / 100.0;
    strengthValueLabel->setText(QString::number(strength, 'f', 2));
    emit strengthChanged(strength);
}
void AIStyleDock::onStepsSliderChanged(int value) {
    stepsValueLabel->setText(QString::number(value));
    emit stepsChanged(value);
}

void AIStyleDock::onPresetClicked() {
    QPushButton *btn = qobject_cast<QPushButton*>(sender());
    if (btn) {
        QString preset = btn->property("preset").toString();
        emit presetSelected(preset);
        btn->setProperty("themeRole", "active");
        btn->style()->unpolish(btn);
        btn->style()->polish(btn);
        QTimer::singleShot(200, [btn]() {
            btn->setProperty("themeRole", QVariant());
            btn->style()->unpolish(btn);
            btn->style()->polish(btn);
        });
    }
}

void AIStyleDock::onRefreshStats() {
    // Poll active source status if needed, mostly pushed from C++ side
}

void AIStyleDock::pulseProgress() {
    static int offset = 0;
    offset = (offset + 5) % 100;
    QString style = QString(R"(
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 %1,
                stop:%2,
                stop:1 %1
            );
            border-radius: 2px;
        }
    )").arg(palette().color(QPalette::Highlight).name())
        .arg(offset / 100.0);
    progressBar->setStyleSheet(style);
}

void AIStyleDock::applyPresetToFilter(const QString &preset) {
    if (!connectedSource) return;
    
    obs_data_t *settings = obs_source_get_settings(connectedSource);
    
    if (preset == "Anime") {
        obs_data_set_string(settings, "style_preset", "Anime");
        obs_data_set_string(settings, "prompt", "anime style, cel shaded, vibrant");
        obs_data_set_double(settings, "strength", 0.7);
        promptEdit->setText("anime style, cel shaded, vibrant");
        strengthSlider->setValue(70);
    } else if (preset == "Cinematic") {
        obs_data_set_string(settings, "style_preset", "Cinematic");
        obs_data_set_string(settings, "prompt", "cinematic lighting, dramatic, 8k");
        obs_data_set_double(settings, "strength", 0.8);
        promptEdit->setText("cinematic lighting, dramatic, 8k");
        strengthSlider->setValue(80);
    } else if (preset == "Upscale") {
        obs_data_set_string(settings, "enhancement_mode", "Upscale");
    }
    
    obs_source_update(connectedSource, settings);
    obs_data_release(settings);
    setStatus("Applied: " + preset);
}

// ===== C-Compatible Functions =====

extern "C" {
    void ai_dock_update_preview(const uint8_t* data, int width, int height) {
        if (g_aiStyleDock) g_aiStyleDock->updatePreview(data, width, height);
    }
    void ai_dock_update_stats(float fps, int dropped, float vram_mb) {
        if (g_aiStyleDock) g_aiStyleDock->updateStats(fps, dropped, vram_mb);
    }
    void ai_dock_update_progress(float progress) {
        if (g_aiStyleDock) g_aiStyleDock->updateProgress(progress);
    }
    void ai_dock_set_status(const char* status) {
        if (g_aiStyleDock && status) g_aiStyleDock->setStatus(QString::fromUtf8(status));
    }
    void ai_dock_connect_source(obs_source_t *source) {
        if (g_aiStyleDock && source) g_aiStyleDock->connectToSource(source);
    }
    void ai_dock_register_source(obs_source_t *source) {
        if (g_aiStyleDock) g_aiStyleDock->registerActiveSource(source);
    }
    void ai_dock_unregister_source(obs_source_t *source) {
        if (g_aiStyleDock) g_aiStyleDock->unregisterActiveSource(source);
    }
}
