/*
1. 윈도우 창 생성이 가능하다.
- 윈도우 창 타이틀 이름 수정 
- 윈도우 창 메뉴바 정리
2. 마우스 우 클릭을 이용해 선 그리기가 가능하다.
3. 마우스 좌 클릭을 이용해 지우기 기능이 가능하다.
4. 메뉴바 기능을 사용 가능하다.
- 새로 만들기, 새 창, 저장 및 열기
- 도형(직사각형, 원형)그리기가 가능하다.
- 선 굵기에 대해서 조정이 가능하다.
5. 도형 그리기 시 더블 버퍼링 작업이 된다.
*/

// MFC_PaintView.cpp: CMFCPaintView 클래스의 구현
#include "pch.h"
#include "framework.h"
#ifndef SHARED_HANDLERS
#include "MFC_Paint.h"
#include <vector>
#endif

#include "MFC_PaintDoc.h"
#include "MFC_PaintView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

IMPLEMENT_DYNCREATE(CMFCPaintView, CView)

BEGIN_MESSAGE_MAP(CMFCPaintView, CView)
    ON_WM_LBUTTONDOWN()
    ON_WM_MOUSEMOVE()
    ON_WM_LBUTTONUP()
    ON_WM_RBUTTONDOWN()
    ON_WM_RBUTTONUP()
    ON_COMMAND(ID_LINE_1, &CMFCPaintView::OnLine1)
    ON_COMMAND(ID_LINE_10, &CMFCPaintView::OnLine10)
    ON_COMMAND(ID_LINE_15, &CMFCPaintView::OnLine15)
    ON_COMMAND(ID_Rect, &CMFCPaintView::OnRect)
    ON_COMMAND(ID_Ellipse, &CMFCPaintView::OnEllipse)
    ON_COMMAND(ID_Line, &CMFCPaintView::OnLine)
    ON_COMMAND(ID_FILE_NEW, &CMFCPaintView::OnFileNew)
    ON_COMMAND(ID_FILE_OPEN, &CMFCPaintView::OnFileOpen)
    ON_COMMAND(ID_FILE_SAVE, &CMFCPaintView::OnFileSave)
    ON_COMMAND(ID_Red, &CMFCPaintView::OnRed)
    ON_COMMAND(ID_Blue, &CMFCPaintView::OnBlue)
    ON_COMMAND(ID_Green, &CMFCPaintView::OnGreen)
    ON_COMMAND(ID_L_Red, &CMFCPaintView::OnLRed)
    ON_COMMAND(ID_L_Blue, &CMFCPaintView::OnLBlue)
    ON_COMMAND(ID_L_Green, &CMFCPaintView::OnLGreen)
    ON_COMMAND(ID_VSave, &CMFCPaintView::OnVsave)
END_MESSAGE_MAP()

CMFCPaintView::CMFCPaintView() noexcept
    : m_isDrawing(false), m_isErasing(false), m_PenWidth(1), m_isRect(false), m_isEllipse(false), m_isLine(true), isRed(false), isBlue(false), isGreen(false), F_isRed(false), F_isBlue(false), F_isGreen(false)
{
}

CMFCPaintView::~CMFCPaintView()
{
    if (m_BackBuffer.GetSafeHandle())
    {
        m_BackBuffer.DeleteObject();
    }

    if (m_BackBufferDC.GetSafeHdc())
    {
        m_BackBufferDC.DeleteDC();
    }
}

BOOL CMFCPaintView::PreCreateWindow(CREATESTRUCT& cs)
{
    return CView::PreCreateWindow(cs);
}

void CMFCPaintView::OnDraw(CDC* pDC)
{
    CMFCPaintDoc* pDoc = GetDocument();
    ASSERT_VALID(pDoc);
    if (!pDoc)
        return;

    if (!m_BackBuffer.GetSafeHandle())
    {
        CRect rect;
        GetClientRect(&rect);
        m_BackBuffer.CreateCompatibleBitmap(pDC, rect.Width(), rect.Height());
        m_BackBufferDC.CreateCompatibleDC(pDC);
        m_BackBufferDC.SelectObject(&m_BackBuffer);
        m_BackBufferDC.FillSolidRect(&rect, RGB(255, 255, 255));
    }

    pDC->BitBlt(0, 0, m_BackBufferDC.GetDeviceCaps(HORZRES), m_BackBufferDC.GetDeviceCaps(VERTRES), &m_BackBufferDC, 0, 0, SRCCOPY);
}

void CMFCPaintView::OnLButtonDown(UINT nFlags, CPoint point)
{
    m_startPoint = point;
    m_isDrawing = true;
    m_isErasing = false;
    CView::OnLButtonDown(nFlags, point);
}

// 선 그릴때 사용 할 벡터
std::vector<CPoint> m_points;

void CMFCPaintView::OnMouseMove(UINT nFlags, CPoint point)
{
    if (m_isDrawing)
    {
        m_endPoint = point;

        // 현재 창 크기를 알아냄.
        CRect rect;
        GetClientRect(&rect);
        CDC tempDC;
        tempDC.CreateCompatibleDC(&m_BackBufferDC);
        CBitmap tempBitmap;
        tempBitmap.CreateCompatibleBitmap(&m_BackBufferDC, rect.Width(), rect.Height());
        tempDC.SelectObject(&tempBitmap);
        tempDC.BitBlt(0, 0, rect.Width(), rect.Height(), &m_BackBufferDC, 0, 0, SRCCOPY);

        CPen pen(PS_SOLID, m_PenWidth, LineColor);
        CPen* pOldPen = tempDC.SelectObject(&pen);

        //CBrush* pOldBrush = tempDC.SelectObject(CBrush::FromHandle((HBRUSH)GetStockObject(NULL_BRUSH)));

        //CBrush brush;
        //brush.CreateStockObject(FullColor);
        //CBrush* pOldBrush = tempDC.SelectObject(&brush);

        CBrush brush;
        //brush.CreateStockObject(NULL_BRUSH);

        if (F_isRed || F_isBlue || F_isGreen)
        {
            brush.CreateSolidBrush(FullColor); // 원하는 색상으로 변경하세요.
        }
        else
        {
            brush.CreateStockObject(NULL_BRUSH);
        }
        CBrush* pOldBrush = tempDC.SelectObject(&brush);

        if (m_isRect)
        {
            tempDC.Rectangle(CRect(m_startPoint, m_endPoint));
        }
        else if (m_isEllipse)
        {
            tempDC.Ellipse(CRect(m_startPoint, m_endPoint));
        }
        else if (m_isLine)
        {
            if (!m_points.empty())
            {
                tempDC.MoveTo(m_points[0]);
                for (size_t i = 1; i < m_points.size(); ++i)
                {
                    tempDC.LineTo(m_points[i]);
                }
            }
            tempDC.LineTo(point);
        }

        tempDC.SelectObject(pOldPen);
        //tempDC.SelectObject(pOldBrush);

        // 임시 버퍼 출력.
        CClientDC dc(this);
        dc.BitBlt(0, 0, rect.Width(), rect.Height(), &tempDC, 0, 0, SRCCOPY);

        // 마우스 위치를 벡터에 추가.
        if (m_isLine)
        {
            m_points.push_back(point);

        }

        CString strFilePath = _T("a.txt");
        CStdioFile file;
        if (file.Open(strFilePath, CFile::modeWrite | CFile::modeCreate | CFile::modeNoTruncate))
        {
            file.SeekToEnd();
            CString strPoint;
            strPoint.Format(_T("%d,%d\n"), point.x, point.y);
            file.WriteString(strPoint);
            file.Close();
        }

    }
    else if (m_isErasing)
    {
        CPen pen(PS_SOLID, m_PenWidth, RGB(255, 255, 255));
        CPen* pOldPen = m_BackBufferDC.SelectObject(&pen);
        m_BackBufferDC.MoveTo(m_ptPrev);
        m_BackBufferDC.LineTo(point);
        m_BackBufferDC.SelectObject(pOldPen);
        m_ptPrev = point;
        Invalidate(FALSE);
    }

    CView::OnMouseMove(nFlags, point);
}

void CMFCPaintView::OnLButtonUp(UINT nFlags, CPoint point)
{
    if (m_isDrawing)
    {
        m_endPoint = point;
        m_isDrawing = false;

        // 백 버퍼에 최종 도형을 그림.
        CPen pen(PS_SOLID, m_PenWidth, LineColor);
        CPen* pOldPen = m_BackBufferDC.SelectObject(&pen);

        //CBrush* pOldBrush = m_BackBufferDC.SelectObject(CBrush::FromHandle((HBRUSH)GetStockObject(NULL_BRUSH)));

        CBrush brush;
        //brush.CreateStockObject(NULL_BRUSH);

        if (F_isRed || F_isBlue || F_isGreen)
        {
            brush.CreateSolidBrush(FullColor); // 원하는 색상으로 변경하세요.
        }
        else
        {
            brush.CreateStockObject(NULL_BRUSH);
        }
        CBrush* pOldBrush = m_BackBufferDC.SelectObject(&brush);

        if (m_isRect)
        {
            m_BackBufferDC.Rectangle(CRect(m_startPoint, m_endPoint));
        }
        else if (m_isEllipse)
        {
            m_BackBufferDC.Ellipse(CRect(m_startPoint, m_endPoint));
        }
        else if (m_isLine)
        {
            if (!m_points.empty())
            {
                m_BackBufferDC.MoveTo(m_points[0]);
                for (size_t i = 1; i < m_points.size(); ++i)
                {
                    m_BackBufferDC.LineTo(m_points[i]);
                }
            }
            m_points.clear();
        }

        m_BackBufferDC.SelectObject(pOldPen);
        m_BackBufferDC.SelectObject(pOldBrush);
        
        Invalidate(FALSE);
    }

    CView::OnLButtonUp(nFlags, point);
}

void CMFCPaintView::OnRButtonDown(UINT nFlags, CPoint point)
{
    m_ptPrev = point;
    m_isErasing = true;
    m_isDrawing = false;
    CView::OnRButtonDown(nFlags, point);
}

void CMFCPaintView::OnRButtonUp(UINT nFlags, CPoint point)
{
    if (m_isErasing)
    {
        m_isErasing = false;
    }
    CView::OnRButtonUp(nFlags, point);
}

// CMFCPaintView 진단
#ifdef _DEBUG
void CMFCPaintView::AssertValid() const
{
    CView::AssertValid();
}

void CMFCPaintView::Dump(CDumpContext& dc) const
{
    CView::Dump(dc);
}

CMFCPaintDoc* CMFCPaintView::GetDocument() const
{
    ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CMFCPaintDoc)));
    return (CMFCPaintDoc*)m_pDocument;
}
#endif //_DEBUG

//************* 선 굵기 *************
void CMFCPaintView::OnLine1()
{
    m_PenWidth = 1;
}

void CMFCPaintView::OnLine10()
{
    m_PenWidth = 10;
}

void CMFCPaintView::OnLine15()
{
    m_PenWidth = 15;
}
//************* 선 굵기 *************

//************** 그리기 **************
void CMFCPaintView::OnRect()
{
    m_isRect = true;
    m_isEllipse = false;
    m_isLine = false;
}

void CMFCPaintView::OnEllipse()
{
    m_isRect = false;
    m_isEllipse = true;
    m_isLine = false;
}

void CMFCPaintView::OnLine()
{
    m_isRect = false;
    m_isEllipse = false;
    m_isLine = true;
}

void CMFCPaintView::OnRed()
{
    F_isRed = true;
    FullColor = RGB(255, 0, 0);
}

void CMFCPaintView::OnBlue()
{
    F_isBlue = true;
    FullColor = RGB(0, 0, 255);
}

void CMFCPaintView::OnGreen()
{
    F_isGreen = true;
    FullColor = RGB(0, 255, 0);
}

void CMFCPaintView::OnLRed()
{
    isRed = true;
    LineColor = RGB(255, 0, 0);
}

void CMFCPaintView::OnLBlue()
{
    isBlue = true;
    LineColor = RGB(0, 0, 255);
}

void CMFCPaintView::OnLGreen()
{
    isGreen = true;
    LineColor = RGB(0, 255, 0);
}

//************** 그리기 **************

//**************** 메뉴 ****************
void CMFCPaintView::OnFileNew()
{
    CRect rect;
    GetClientRect(&rect);
    m_BackBufferDC.FillSolidRect(&rect, RGB(255, 255, 255));
    Invalidate(FALSE);
}

void CMFCPaintView::OnFileOpen()
{
    CFileDialog dlg(TRUE, _T("bmp"), NULL, OFN_FILEMUSTEXIST | OFN_HIDEREADONLY, _T("Image Files|*.bmp;*.jpg;*.png||"), this);
    if (dlg.DoModal() == IDOK)
    {
        CString strFilePath = dlg.GetPathName();

        // CImage를 사용하여 이미지 파일 로드
        CImage image;
        if (SUCCEEDED(image.Load(strFilePath)))
        {
            // 백 버퍼 DC에 이미지 그리기
            CRect rect;
            GetClientRect(&rect);

            int nWidth = image.GetWidth();
            int nHeight = image.GetHeight();
     
            // 이미지를 백 버퍼 DC에 그리기
            image.Draw(m_BackBufferDC, 0, 0, nWidth, nHeight);

            Invalidate(FALSE);
        }
        else
        {
            // 이미지 로드 실패 시 오류 메시지 표시
            AfxMessageBox(_T("Failed to load the image file."));
        }
    }
}

void CMFCPaintView::OnFileSave()
{
    CFileDialog dlg(FALSE, _T("bmp"), NULL, OFN_OVERWRITEPROMPT, _T("Bitmap Files (*.bmp)|*.bmp||"), this);
    if (dlg.DoModal() == IDOK)
    {
        CString strFilePath = dlg.GetPathName();

        // 백 버퍼의 내용을 비트맵으로 저장
        CRect rect;
        GetClientRect(&rect);
        CBitmap bitmap;
        bitmap.CreateCompatibleBitmap(&m_BackBufferDC, rect.Width(), rect.Height());
        CDC memDC;
        memDC.CreateCompatibleDC(&m_BackBufferDC);
        CBitmap* pOldBitmap = memDC.SelectObject(&bitmap);
        memDC.BitBlt(0, 0, rect.Width(), rect.Height(), &m_BackBufferDC, 0, 0, SRCCOPY);
        memDC.SelectObject(pOldBitmap);

        // 비트맵을 파일로 저장
        CImage image;
        image.Attach(bitmap);
        image.Save(strFilePath);
        image.Detach();
    }
}

void CMFCPaintView::OnVsave()
{
    CString strFilePath = _T("a.txt");
    CStdioFile file;
    if (file.Open(strFilePath, CFile::modeRead))
    {
        CClientDC dc(this);
        CPen pen(PS_SOLID, m_PenWidth, LineColor);
        CPen* pOldPen = dc.SelectObject(&pen);

        CString strLine;
        while (file.ReadString(strLine))
        {
            int commaPos = strLine.Find(',');
            if (commaPos != -1)
            {
                int x = _ttoi(strLine.Left(commaPos));
                int y = _ttoi(strLine.Mid(commaPos + 1));

                if (dc.m_hDC != NULL)
                {
                    if (m_points.empty())
                    {
                        dc.MoveTo(x, y);
                    }
                    else
                    {
                        dc.LineTo(x, y);
                    }
                }

                m_points.push_back(CPoint(x, y));
            }
        }

        dc.SelectObject(pOldPen);
        file.Close();
    }
}
//**************** 메뉴 ****************
