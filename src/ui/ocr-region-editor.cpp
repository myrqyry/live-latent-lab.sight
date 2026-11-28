#include "ui/ocr-region-editor.hpp"
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QSpinBox>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
#include <algorithm>

namespace sight {
namespace {
class Canvas final: public QWidget {
public:
 EditorRegions regions;
 int slot=0;
 QImage frame;
 std::function<void(OcrEditorRect)> selected;
 explicit Canvas(QWidget* parent):QWidget(parent){
    setObjectName("sightOcrCanvas");setMinimumSize(360,220);
    setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
    setCursor(Qt::CrossCursor);
 }
 OcrEditorFit fit() const{return ocr_editor_fit(width(),height(),frame.width(),frame.height());}
 void setFrame(QImage image){frame=std::move(image);drag=false;update();}
protected:
 void paintEvent(QPaintEvent*) override{
    QPainter p(this);p.fillRect(rect(),palette().brush(QPalette::Base));
    if(frame.isNull()){
        p.setPen(palette().color(QPalette::PlaceholderText));
        p.drawText(rect(),Qt::AlignCenter,tr("Waiting for a source frame…"));return;
    }
    auto f=fit();QRectF area(f.x,f.y,f.w,f.h);
    p.setRenderHint(QPainter::SmoothPixmapTransform);p.drawImage(area,frame);
    auto display=[&](OcrEditorRect r){
        r=ocr_editor_clamp(r);
        return QRectF(f.x+f.w*r.x/100.,f.y+f.h*r.y/100.,f.w*r.w/100.,f.h*r.h/100.);
    };
    p.save();p.setClipRect(area);
    for(int i=0;i<4;++i){
        if(!regions[i].enabled&&i!=slot)continue;
        auto r=regions[i].rect;
        if(i==slot&&drag)if(auto pending=ocr_editor_drag(f,start.x(),start.y(),end.x(),end.y()))r=*pending;
        auto box=display(r);
        QColor ink=palette().color(i==slot?QPalette::Highlight:QPalette::Link);
        QColor shade=ink;shade.setAlpha(i==slot?48:20);
        p.fillRect(box,shade);
        // Contrasting under-stroke keeps the border legible over any source.
        p.setPen(QPen(Qt::black,3));p.drawRect(box);
        p.setPen(QPen(ink,i==slot?2:1,regions[i].enabled?Qt::SolidLine:Qt::DashLine));p.drawRect(box);
        QString label=QString::number(i+1)+" · "+regions[i].name;
        QFontMetrics metrics(p.font());
        auto text=metrics.elidedText(label,Qt::ElideRight,std::max(16,int(area.width())-12));
        QRectF badge(box.left(),box.top(),metrics.horizontalAdvance(text)+10,metrics.height()+6);
        badge.moveLeft(std::clamp(badge.left(),area.left(),std::max(area.left(),area.right()-badge.width())));
        badge.moveTop(std::clamp(badge.top(),area.top(),std::max(area.top(),area.bottom()-badge.height())));
        p.fillRect(badge,palette().brush(i==slot?QPalette::Highlight:QPalette::Base));
        p.setPen(palette().color(i==slot?QPalette::HighlightedText:QPalette::Text));
        p.drawText(badge.adjusted(5,0,-5,0),Qt::AlignVCenter,text);
    }
    p.restore();
 }
 void mousePressEvent(QMouseEvent* e) override{
    auto f=fit();auto pos=e->position();
    if(e->button()==Qt::LeftButton&&f.w>0&&f.h>0&&QRectF(f.x,f.y,f.w,f.h).contains(pos)){
        start=end=pos;drag=true;update();e->accept();
    }
 }
 void mouseMoveEvent(QMouseEvent* e) override{if(drag){end=e->position();update();}}
 void mouseReleaseEvent(QMouseEvent* e) override{
    if(!drag||e->button()!=Qt::LeftButton)return;
    drag=false;
    if(auto r=ocr_editor_drag(fit(),start.x(),start.y(),e->position().x(),e->position().y()))
        if(selected)selected(*r);
    update();
 }
private:
 bool drag=false;QPointF start,end;
};
}
std::optional<EditorRegions> edit_ocr_regions(QWidget* parent,EditorRegions initial,
    unsigned sw,unsigned sh,EditorRequest request,EditorRead read){
 // Keep this stack-owned modal independent of a parent that OBS could
 // destroy during a scene collection change or application shutdown.
 QDialog dialog;
 dialog.setWindowModality(Qt::ApplicationModal);
 if(parent){dialog.setPalette(parent->palette());dialog.setFont(parent->font());}
 dialog.setObjectName("sightOcrRegionEditor");dialog.setWindowTitle(QObject::tr("lll.sight · OCR regions"));
 dialog.resize(900,620);
 auto* root=new QVBoxLayout(&dialog);
 auto* instructions=new QLabel(QObject::tr("Select a region, then drag a rectangle over the source snapshot. Changes apply only when you choose OK."),&dialog);
 instructions->setWordWrap(true);root->addWidget(instructions);
 auto* top=new QHBoxLayout;
 auto* slot=new QComboBox(&dialog);slot->setObjectName("sightOcrSlot");
 for(int i=0;i<4;++i)slot->addItem(QObject::tr("Region %1").arg(i+1));
 auto* enabled=new QCheckBox(QObject::tr("Enabled"),&dialog);enabled->setObjectName("sightOcrEnabled");
 auto* name=new QLineEdit(&dialog);name->setObjectName("sightOcrName");name->setMaxLength(128);name->setPlaceholderText(QObject::tr("Region name"));
 top->addWidget(slot);top->addWidget(enabled);top->addWidget(name,1);
 auto* refresh=new QPushButton(QObject::tr("Refresh snapshot"),&dialog);refresh->setObjectName("sightOcrRefresh");
 top->addWidget(refresh);root->addLayout(top);
 auto* canvas=new Canvas(&dialog);canvas->regions=initial;
 for(auto& r:canvas->regions)r.rect=ocr_editor_clamp(r.rect);
 canvas->regions[0].enabled=true;
 root->addWidget(canvas,1);
 auto* coordinates=new QHBoxLayout;
 std::array<QSpinBox*,4> spins{};
 const std::array<const char*,4> labels{"Left","Top","Width","Height"};
 for(int i=0;i<4;++i){
    coordinates->addWidget(new QLabel(QObject::tr(labels[i]),&dialog));
    spins[i]=new QSpinBox(&dialog);spins[i]->setObjectName(QString("sightOcrCoord%1").arg(i));
    spins[i]->setRange(i<2?0:1,i<2?99:100);spins[i]->setSuffix("%");coordinates->addWidget(spins[i]);
 }
 auto* reset=new QPushButton(QObject::tr("Full frame"),&dialog);coordinates->addWidget(reset);root->addLayout(coordinates);
 auto* status=new QLabel(QObject::tr("Waiting for the existing OCR capture…"),&dialog);
 status->setObjectName("sightOcrStatus");status->setWordWrap(true);root->addWidget(status);
 auto* buttons=new QDialogButtonBox(QDialogButtonBox::Ok|QDialogButtonBox::Cancel,&dialog);root->addWidget(buttons);
 auto* ok=buttons->button(QDialogButtonBox::Ok);ok->setEnabled(false);
 bool loading=false,has_frame=false;
 auto sync=[&](){
    loading=true;int i=slot->currentIndex();canvas->slot=i;const auto& r=canvas->regions[i];
    enabled->setChecked(r.enabled);enabled->setEnabled(i!=0);name->setText(r.name);
    spins[0]->setValue(r.rect.x);spins[1]->setValue(r.rect.y);spins[2]->setValue(r.rect.w);spins[3]->setValue(r.rect.h);
    loading=false;canvas->update();
 };
 canvas->selected=[&](OcrEditorRect r){auto& current=canvas->regions[canvas->slot];current.rect=r;current.enabled=true;sync();};
 QObject::connect(slot,qOverload<int>(&QComboBox::currentIndexChanged),&dialog,[&](int){sync();});
 QObject::connect(enabled,&QCheckBox::toggled,&dialog,[&](bool on){if(!loading){canvas->regions[canvas->slot].enabled=on;canvas->update();}});
 QObject::connect(name,&QLineEdit::textEdited,&dialog,[&](const QString& text){
    if(text.toUtf8().size()>128){
        name->setText(canvas->regions[canvas->slot].name);return;
    }
    canvas->regions[canvas->slot].name=text;canvas->update();
 });
 for(auto* spin:spins)QObject::connect(spin,qOverload<int>(&QSpinBox::valueChanged),&dialog,[&](int){
    if(loading)return;
    canvas->regions[canvas->slot].rect=ocr_editor_clamp({spins[0]->value(),spins[1]->value(),spins[2]->value(),spins[3]->value()});sync();
 });
 QObject::connect(reset,&QPushButton::clicked,&dialog,[&](){canvas->regions[canvas->slot].rect={};sync();});
 QObject::connect(buttons,&QDialogButtonBox::accepted,&dialog,&QDialog::accept);
 QObject::connect(buttons,&QDialogButtonBox::rejected,&dialog,&QDialog::reject);
 QTimer timer;timer.setInterval(75);
 int attempts=0;
 auto start=[&](){
    has_frame=false;ok->setEnabled(false);canvas->setFrame({});attempts=0;
    status->setText(QObject::tr("Waiting for the existing OCR capture…"));request();timer.start();
 };
 QObject::connect(refresh,&QPushButton::clicked,&dialog,start);
 QObject::connect(&timer,&QTimer::timeout,&dialog,[&](){
    auto frame=read();
    if(frame&&!frame->image.isNull()){
        timer.stop();
        if(frame->source_width!=sw||frame->source_height!=sh){
            status->setText(QObject::tr("Source dimensions changed. Cancel and reopen the editor."));return;
        }
        canvas->setFrame(std::move(frame->image));has_frame=true;ok->setEnabled(true);
        status->setText(QObject::tr("Frozen source snapshot · %1 × %2. Refresh for a newer frame.").arg(sw).arg(sh));
    }else if(++attempts>=160){
        timer.stop();status->setText(QObject::tr("No frame received. Make the source visible and enable the filter, then refresh."));
    }
 });
 sync();start();
 if(dialog.exec()!=QDialog::Accepted||!has_frame)return {};
 return canvas->regions;
}
}
